import torch
import torch.nn as nn
from pointpillars.pointpillars import PillarFeatureNet, PointPillarsScatter


class PillarOfFafe(nn.Module):
    def __init__(self, input_config, batch_size, verbose):
        super().__init__()
        self.verbose = verbose
        self.feature_net = PillarFeatureNet(num_input_features=input_config.num_point_features,
                                            use_norm=input_config.pp_use_norm,
                                            num_filters=input_config.pp_num_filters,
                                            with_distance=input_config.pp_with_distance,
                                            voxel_size=(input_config.x_grid_size, input_config.x_grid_size,
                                                        input_config.z_pillar_size),
                                            pc_range=(input_config.x_min, input_config.y_min, input_config.z_min,
                                                      input_config.x_max, input_config.y_max, input_config.z_max),
                                            verbose=verbose)
        # output_shape = (X, X, nx, ny)

        self.psuedo_creator = PointPillarsScatter(input_config, num_input_features=input_config.pp_num_filters[-1],
                                                  verbose=verbose)
        self._batch_size = batch_size

    def forward(self, voxels, num_points, coordinates, num_nonempty_voxels):
        """
        The voxels from different batches are the same size as input, but later in this function
        num_nonempty_voxels is used to extract only nonempty information.
        :param voxels: shape [nB, nMaxVoxels, nMaxPoints, nPointFeatures]
        :param num_points:  [nB, nMaxVoxels]
        :param coordinates: [nB, nMaxVoxels, 3]
        :param num_nonempty_voxels: [nB]
        :return:
        """

        pseudos = []

        # Checks the case if there are not enough frames in one sequence to build enough to match batch size.
        if voxels.shape[0] < self._batch_size:
            batch_size = voxels.shape[0]
        else:
            batch_size = self._batch_size

        for batch_iter in range(batch_size):
            spatial_features = self.feature_net(voxels[batch_iter][:num_nonempty_voxels[batch_iter].int()],
                                                num_points[batch_iter][:num_nonempty_voxels[batch_iter].int()],
                                                coordinates[batch_iter][:num_nonempty_voxels[batch_iter].int()])

            pseudo = self.psuedo_creator(spatial_features, coordinates[batch_iter][:num_nonempty_voxels[batch_iter].int()])
            pseudos.append(pseudo.unsqueeze(0))

        pseudo_image = torch.cat(pseudos, dim=0)
        return pseudo_image
