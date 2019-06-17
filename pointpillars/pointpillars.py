"""
PointPillars fork from SECOND.
Code written by Alex Lang and Oscar Beijbom, 2018.
Licensed under MIT License [see LICENSE].
"""

import torch
from torch import nn
from torch.nn import functional as F

from pointpillars.utils import get_paddings_indicator, Empty, change_default_args


class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        """
        Pillar Feature Net Layer.
        The Pillar Feature Net could be composed of a series of these layers, but the PointPillars paper results only
        used a single PFNLayer. This layer performs a similar role as second.pytorch.voxelnet.VFELayer.
        :param in_channels: <int>. Number of input channels.
        :param out_channels: <int>. Number of output channels.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param last_layer: <bool>. If last_layer, there is no concatenation of features.
        """

        super().__init__()
        self.name = 'PFNLayer'
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels

        if use_norm:
            BatchNorm1d = change_default_args(eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Linear = change_default_args(bias=False)(nn.Linear)
        else:
            BatchNorm1d = Empty
            Linear = change_default_args(bias=True)(nn.Linear)

        self.linear = Linear(in_channels, self.units)
        self.norm = BatchNorm1d(self.units)

    def forward(self, inputs):

        x = self.linear(inputs)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        x = F.relu(x)

        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class PillarFeatureNet(nn.Module):
    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=(64,),
                 with_distance=False,
                 voxel_size=(0.2, 0.2, 4),
                 pc_range=(0, -40, -3, 70.4, 40, 1),
                 verbose=False):
        """
        Pillar Feature Net.
        The network prepares the pillar features and performs forward pass through PFNLayers. This net performs a
        similar role to SECOND's second.pytorch.voxelnet.VoxelFeatureExtractor.
        :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param num_filters: (<int>: N). Number of features in each of the N PFNLayers.
        :param with_distance: <bool>. Whether to include Euclidean distance to points.
        :param voxel_size: (<float>: 3). Size of voxels, only utilize x and y size.
        :param pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
        """

        super().__init__()
        self.name = 'PillarFeatureNet'
        assert len(num_filters) > 0
        num_input_features += 5
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance

        # Create PillarFeatureNet layers
        num_filters = [num_input_features] + list(num_filters)
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(PFNLayer(in_filters, out_filters, use_norm, last_layer=last_layer))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate pillar offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + pc_range[0]
        self.y_offset = self.vy / 2 + pc_range[1]
        self.verbose = verbose

    def forward(self, voxels, num_points, coors):
        """

        :param voxels:      - [nNonEmptyVoxels, max_points, ndim] float tensor. only contain points.
        :param num_points:  - num_points_per_voxel: [nNonEmptyVoxels] int32 tensor.
        :param coors:       - coordinates: [nNonEmptyVoxels, 3] int32 tensor
        :return:
        """

        # Find distance of x, y, and z from cluster center
        points_mean = voxels[:, :, :3].sum(dim=1, keepdim=True) / num_points.type_as(voxels).view(-1, 1, 1)
        f_cluster = voxels[:, :, :3] - points_mean

        # Find distance of x, y, and z from pillar center
        f_center = voxels[:, :, :2]
        f_center[:, :, 0] = f_center[:, :, 0] - (coors[:, 0].float().unsqueeze(1) * self.vx + self.x_offset)
        f_center[:, :, 1] = f_center[:, :, 1] - (coors[:, 1].float().unsqueeze(1) * self.vy + self.y_offset)

        # Combine together feature decorations
        features_ls = [voxels, f_cluster, f_center]
        if self._with_distance:
            points_dist = torch.norm(voxels[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)
        voxels = torch.cat(features_ls, dim=-1)

        # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = voxels.shape[1]
        mask = get_paddings_indicator(num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxels)
        voxels *= mask

        # Forward pass through PFNLayers
        for pfn in self.pfn_layers:
            voxels = pfn(voxels)

        return voxels.squeeze()


class PointPillarsScatter(nn.Module):

    def __init__(self,
                 input_config,
                 num_input_features=64,
                 verbose=False):
        """
        Point Pillar's Scatter.
        Converts learned features from dense tensor to sparse pseudo image. This replaces SECOND's
        second.pytorch.voxelnet.SparseMiddleExtractor.
        :param output_shape: ([int]: 4). Required output shape of features.
        :param num_input_features: <int>. Number of input features.
        """

        super().__init__()
        self.name = 'PointPillarsScatter'
        self.nx = input_config.bev_shape[0]
        self.ny = input_config.bev_shape[1]
        self.nchannels = num_input_features
        self.verbose = verbose

    def forward(self, voxel_features, coords):
        """

        :param voxel_features:  [nNonEmptyVoxels, nChannels]
        :param coords:          [nNonEmptyVoxels, 3]
        :return: pseudo_image   [nChannels, nX, nY]
        """
        # Create the canvas for this sample
        canvas = torch.zeros(self.nchannels, self.nx * self.ny, dtype=voxel_features.dtype,
                             device=voxel_features.device)
        this_coords = coords.type(torch.long)
        indices = this_coords[:, 0] + this_coords[:, 1] * self.nx
        indices = indices.type(torch.long)
        voxels = voxel_features.t()

        # Now scatter the blob back to the canvas.
        canvas[:, indices] = voxels

        # Undo the column stacking to final 4-dim tensor

        return canvas.view(self.nchannels, self.nx, self.ny)
