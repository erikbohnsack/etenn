import numpy as np
from pointpillars.utils import points_to_voxel


class VoxelGenerator:
    """
    VoxelGenerator discretizes the raw point cloud to voxels.
    """

    def __init__(self,
                 voxel_size,
                 point_cloud_range,
                 max_num_points,
                 reverse_index,
                 max_voxels=20000):
        point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        # [0, -40, -3, 70.4, 40, 1]
        voxel_size = np.array(voxel_size, dtype=np.float32)
        grid_size = (
                            point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
        grid_size = np.round(grid_size).astype(np.int64)

        self._voxel_size = voxel_size
        self._point_cloud_range = point_cloud_range
        self._max_num_points = max_num_points
        self._max_voxels = max_voxels
        self._grid_size = grid_size
        self._reverse_index = reverse_index

    def generate(self, points):
        """

        :param      - points: [N, ndim] float tensor. points[:, :3] contain xyz points and
            points[:, 3:] contain other information such as reflectivity.

        :return:    - voxels: [M, max_points, ndim] float tensor. only contain points.
                    - coordinates: [M, 3] int32 tensor.
                    - num_points_per_voxel: [M] int32 tensor.
        """
        return points_to_voxel(
            points, voxel_size=self._voxel_size, coors_range=self._point_cloud_range,
            max_points=self._max_num_points, reverse_index=self._reverse_index, max_voxels=self._max_voxels)

    @property
    def voxel_size(self):
        return self._voxel_size

    @property
    def max_num_points_per_voxel(self):
        return self._max_num_points

    @property
    def point_cloud_range(self):
        return self._point_cloud_range

    @property
    def grid_size(self):
        return self._grid_size
