import numpy as np
import torch
import os
from torch.utils.data import Dataset, Sampler
from fafe_utils.label import get_labels, get_labels_car, reshape_labels
from fafe_utils.fafe_utils import format_to_BEV_jit
from pointpillars.voxelgenerator import VoxelGenerator


class FafeSampler(Sampler):
    """
    Sampler that does not fuck around with taking too early indices.
    """

    def __init__(self, data_source, input_config):
        self.config = input_config
        self.data_source = data_source
        super().__init__(data_source)

    def __iter__(self):
        return iter([i for i in range(len(self.data_source)) if i >= self.config.num_conseq_frames - 1 and
                     len(self.data_source) - i > self.config.num_conseq_frames])

    def __len__(self):
        len(self.data_source)


class RawDataset(Dataset):
    def __init__(self, input_config, root_dir, split, sequence):
        self.data_path = os.path.join(root_dir, split, "velodyne", str(sequence).zfill(4))
        self.sequence = sequence
        self.num_point_features = input_config.num_point_features
        self.config = input_config

    def __len__(self):
        return len(os.listdir(os.path.join(self.data_path)))

    def __getitem__(self, frame):
        file = os.path.join(self.data_path, str(frame).zfill(6) + '.bin')
        points = np.fromfile(file, dtype=np.float32).reshape((-1, 4))
        return torch.from_numpy(points)


class VoxelDataset(Dataset):
    def __init__(self, input_config, root_dir, split, sequence):
        self.data_path = os.path.join(root_dir, split, "velodyne", str(sequence).zfill(4))
        self.sequence = sequence
        self.num_point_features = input_config.num_point_features
        self.config = input_config
        self.num_conseq_frames = input_config.num_conseq_frames
        self.voxel_generator = VoxelGenerator(
            voxel_size=(input_config.x_grid_size, input_config.x_grid_size, input_config.z_pillar_size),
            point_cloud_range=(input_config.x_min, input_config.y_min, input_config.z_min,
                               input_config.x_max, input_config.y_max, input_config.z_max),
            max_num_points=input_config.pp_max_points,
            reverse_index=input_config.pp_reverse_index,
            max_voxels=input_config.pp_max_voxels)

        label_path = os.path.join(root_dir, split, 'label_2', str(self.sequence).zfill(4) + '.txt')

        if input_config.get_labels == 'car':
            self._labels_dict, max_frame_idx = get_labels_car(label_path)
            print('\tSequence {} [{} frames] <get_labels_car>'.format(self.sequence, max_frame_idx))
        else:
            self._labels_dict, max_frame_idx = get_labels(label_path)
            print('\tSequence {} [{} frames] <get_labels>'.format(self.sequence, max_frame_idx))

        self.labels = reshape_labels(self._labels_dict, input_config)

    def __len__(self):
        return len(os.listdir(os.path.join(self.data_path))) - (self.num_conseq_frames - 1) * 2

    def __getitem__(self, index):
        real_index = int(index + self.num_conseq_frames - 1)

        input_indices = [x for x in range(real_index - self.num_conseq_frames + 1, real_index + 1)]
        output_indices = [x for x in range(real_index, real_index + self.num_conseq_frames)]

        info = {}
        info['Current_index'] = real_index
        info['BEV_indices'] = input_indices
        info['GT_indices'] = output_indices
        info['sequence'] = self.sequence

        voxel_stack = torch.zeros(self.num_conseq_frames, self.config.pp_max_voxels, self.config.pp_max_points,
                                  self.config.num_point_features)

        coord_stack = torch.zeros(self.num_conseq_frames, self.config.pp_max_voxels, 3)

        num_points_stack = torch.zeros(self.num_conseq_frames, self.config.pp_max_voxels)
        num_nonempty_voxels = torch.zeros(self.num_conseq_frames)

        for i, frame in enumerate(input_indices):

            file = os.path.join(self.data_path, str(frame).zfill(6) + '.bin')
            points = np.fromfile(file, dtype=np.float32).reshape((-1, 4))

            if self.num_point_features == 3:
                points = points[:, :3]
            else:
                assert self.num_point_features == 4
            voxels, coordinates, num_points = self.voxel_generator.generate(points)

            voxel_stack[i, :voxels.shape[0]] = torch.from_numpy(voxels)
            coord_stack[i, :voxels.shape[0]] = torch.from_numpy(coordinates)
            num_points_stack[i, :voxels.shape[0]] = torch.from_numpy(num_points)
            num_nonempty_voxels[i] = voxels.shape[0]

        try:
            target = self.labels[output_indices, ...]
        except Exception as e:
            print('Tried to get: {} \n from sequence {}'.format(output_indices, self.sequence))
            raise e
        return voxel_stack, coord_stack, num_points_stack, num_nonempty_voxels, target, info


class KittiDataset(Dataset):
    """ KITTI DATASET """

    def __init__(self, input_config, root_dir, split, sequence):
        """

        :param root_dir: path to data
        :param split: str, either "training" or "testing"
        """
        self.data_path = os.path.join(root_dir, split, "velodyne", str(sequence).zfill(4))
        self.sequence = sequence
        self.num_point_features = input_config.num_point_features
        self.config = input_config

    def __len__(self):
        return len(os.listdir(os.path.join(self.data_path)))

    def __getitem__(self, index):
        """

        :param index: frame idx
        :return: Torch.Tensor of stacked temporal BEV maps.
        """
        assert index >= self.config.num_conseq_frames - 1

        grid_limits = np.array([[self.config.x_min, self.config.x_max],
                                [self.config.y_min, self.config.y_max],
                                [self.config.z_min, self.config.z_max]])
        grid_sizes = np.array([self.config.x_grid_size, self.config.y_grid_size, self.config.z_grid_size])

        bev_maps = []
        for i in range(self.config.num_conseq_frames):
            frame = index + 1 - (self.config.num_conseq_frames - i)
            file = os.path.join(self.data_path, str(frame).zfill(6) + '.bin')
            points = np.fromfile(file, dtype=np.float32).reshape((-1, 4))
            bev_map = np.zeros(self.config.bev_shape)
            format_to_BEV_jit(bev_map, points, grid_limits, grid_sizes)
            bev_maps.append(torch.from_numpy(bev_map))
        out = torch.cat(bev_maps, 2)
        return out


class TemporalBEVsDataset(Dataset):
    """ Temporal BEVs DATASET, created with create_bevs.py """

    def __init__(self, input_config, root_dir, split, sequence):
        """

        :param root_dir:
        :param split: str, either "training" or "testing"
        :param sequence: str or integer deciding which sequence to load
        """
        if not isinstance(sequence, str):
            self._sequence = str(sequence).zfill(4)
        else:
            self._sequence = sequence

        self.data_path = os.path.join(root_dir, split, 'velodyne', self._sequence)
        self.num_conseq_frames = input_config.num_conseq_frames
        self.bev_shape = input_config.bev_shape
        self.grid_limits = np.array([[input_config.x_min, input_config.x_max],
                                     [input_config.y_min, input_config.y_max],
                                     [input_config.z_min, input_config.z_max]])
        self.grid_sizes = np.array([input_config.x_grid_size, input_config.y_grid_size, input_config.z_grid_size])

        label_path = os.path.join(root_dir, split, 'label_2', self._sequence + '.txt')

        if input_config.get_labels == 'car':
            self._labels_dict, max_frame_idx = get_labels_car(label_path)
            print('\tSequence {} [{} frames] <get_labels_car>'.format(self._sequence, max_frame_idx))
        else:
            self._labels_dict, max_frame_idx = get_labels(label_path)
            print('\tSequence {} [{} frames] <get_labels>'.format(self._sequence, max_frame_idx))

        self.labels = reshape_labels(self._labels_dict, input_config)

    def __len__(self):
        return len(os.listdir(self.data_path)) - (self.num_conseq_frames - 1) * 2

    def __getitem__(self, index):
        """

        :param index: frame index
        :return:
        """

        real_index = int(index + self.num_conseq_frames - 1)

        input_indices = [x for x in range(real_index - self.num_conseq_frames + 1, real_index + 1)]
        output_indices = [x for x in range(real_index, real_index + self.num_conseq_frames)]

        info = {}
        info['Current_index'] = real_index
        info['BEV_indices'] = input_indices
        info['GT_indices'] = output_indices
        info['sequence'] = self._sequence

        conc_bev_map = np.zeros((self.bev_shape[0], self.bev_shape[1], self.bev_shape[2] * self.num_conseq_frames))
        for i, frame in enumerate(input_indices):
            file = os.path.join(self.data_path, str(frame).zfill(6) + '.bin')
            points = np.fromfile(file, dtype=np.float32).reshape((-1, 4))
            bev_map = np.zeros(self.bev_shape, dtype=float)

            format_to_BEV_jit(bev_map,
                              points,
                              self.grid_limits,
                              self.grid_sizes)
            conc_bev_map[:, :, i * self.bev_shape[2]: (i + 1) * self.bev_shape[2]] = bev_map

        input = torch.from_numpy(conc_bev_map).permute(2, 0, 1).float()
        try:
            target = self.labels[output_indices, ...]
        except Exception as e:
            print('Tried to get: {} \n from sequence {}'.format(output_indices, self._sequence))
            raise e
        return input, target, info


class TestDataset(Dataset):
    def __init__(self, input_config, root_dir, split, sequence):
        """

                :param root_dir:
                :param split: str, either "training" or "testing"
                :param sequence: str or integer deciding which sequence to load
                """
        if not isinstance(sequence, str):
            self._sequence = str(sequence).zfill(4)
        else:
            self._sequence = sequence

        self.data_path = os.path.join(root_dir, split, 'velodyne', self._sequence)
        self.num_conseq_frames = input_config.num_conseq_frames
        self.bev_shape = input_config.bev_shape
        self.grid_limits = np.array([[input_config.x_min, input_config.x_max],
                                     [input_config.y_min, input_config.y_max],
                                     [input_config.z_min, input_config.z_max]])
        self.grid_sizes = np.array([input_config.x_grid_size, input_config.y_grid_size, input_config.z_grid_size])

    def __len__(self):
        return len(os.listdir(self.data_path)) - (self.num_conseq_frames - 1) * 2

    def __getitem__(self, index):
        """

                :param index: frame index
                :return:
                """
        real_index = int(index + self.num_conseq_frames - 1)

        filepath = os.path.join(self.data_path, str(real_index).zfill(6) + '.bin')

        input_indeces = [x for x in range(real_index - self.num_conseq_frames + 1, real_index + 1)]
        output_indeces = [x for x in range(real_index, real_index + self.num_conseq_frames)]

        info = {}
        info['Current_index'] = real_index
        info['BEV_indeces'] = input_indeces
        info['GT_indeces'] = output_indeces
        info['sequence'] = self._sequence

        bev_maps = []
        for frame in input_indeces:
            file = os.path.join(self.data_path, str(frame).zfill(6) + '.bin')
            points = np.fromfile(file, dtype=np.float32).reshape((-1, 4))
            bev_map = np.zeros(self.bev_shape, dtype=float)

            format_to_BEV_jit(bev_map,
                              points,
                              self.grid_limits,
                              self.grid_sizes)

            bev_maps.append(bev_map)

        conc_bev_map = np.concatenate((bev_maps[:]), 2)
        assert np.shape(conc_bev_map)[2] == self.num_conseq_frames * self.bev_shape[2]

        input = torch.from_numpy(conc_bev_map).permute(2, 0, 1).float()

        return input, info
