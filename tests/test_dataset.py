import unittest
import numpy as np
from cfg.config import InputConfig
from fafe_utils import kitti_dataset
from torch.utils.data import DataLoader
import torch
import time
import os
from cfg.config_stuff import load_config, get_showroom_path, get_root_dir
from cfg.config import InputConfig, EvalConfig
import sys



class TestDataLoader(unittest.TestCase):
    def test_voxel(self):
        config_path = '../cfg/cfg_mac.yml'
        config = load_config(config_path)

        input_config = InputConfig(config['INPUT_CONFIG'])
        root_dir = get_root_dir()
        dataset = kitti_dataset.VoxelDataset(input_config, root_dir=root_dir, split='training', sequence=0)

        testing_loader = DataLoader(dataset=dataset, batch_size=1,
                                    num_workers=1)
        for iteration, test_iter in enumerate(testing_loader):
            print("iteration {}".format(iteration))
            voxels, coords, num_points = test_iter


    def test_format_to_BEV(self):
        sys.path.append("..")
        config_path = 'cfg/cfg_mac.yml'
        config = load_config(config_path)

        input_config = InputConfig(config['INPUT_CONFIG'])

        number_of_points = 100
        points = 5 * np.random.rand(number_of_points, input_config.num_point_features) - 1
        bev_map = torch.zeros(input_config.bev_shape)

        kitti_dataset.format_to_BEV(bev_map, points, input_config)
        assert bev_map.shape[0] == (input_config.x_max - input_config.x_min)/input_config.x_grid_size
        assert bev_map.shape[1] == (input_config.y_max - input_config.y_min)/input_config.y_grid_size

        # Assert only 90 out of 100 points since there is a chance they will be in the same
        # "voxel".
        assert sum(sum(sum(bev_map))) > 90

    def test_load_pcd(self):
        sys.path.append("..")
        config_path = 'cfg/cfg_mac.yml'
        config = load_config(config_path)

        input_config = InputConfig(config['INPUT_CONFIG'])
        basepath = "/Users/erikbohnsack/data/training"

        file = os.path.join(basepath, 'velodyne', str(0).zfill(4), str(0).zfill(6) + '.bin')
        points = np.fromfile(file, dtype=np.float32).reshape((-1, 4))
        bev_map = torch.zeros(input_config.bev_shape)

        kitti_dataset.format_to_BEV(bev_map, points, input_config)

        #assert

    def test_temporal_bevs(self):
        sys.path.append("..")
        config_path = 'cfg/cfg_mac.yml'
        config = load_config(config_path)

        input_config = InputConfig(config['INPUT_CONFIG'])
        basepath = "/Users/erikbohnsack/data/training"

        file = os.path.join(basepath, 'velodyne', str(0).zfill(4), str(0).zfill(6) + '.bin')
        points = np.fromfile(file, dtype=np.float32).reshape((-1, 4))
        bev_map = np.zeros(input_config.bev_shape)
        print("BEV map size: {}".format(bev_map.shape))

        start = time.time()
        kitti_dataset.format_to_BEV(bev_map, points, input_config)
        end = time.time()
        print("Elapsed time: {}".format(end - start))

        grid_limits = np.array([[input_config.x_min, input_config.x_max],
                                [input_config.y_min, input_config.y_max],
                                [input_config.z_min, input_config.z_max]])
        grid_sizes = np.array([input_config.x_grid_size, input_config.y_grid_size, input_config.z_grid_size])

        start_jit = time.time()
        kitti_dataset.format_to_BEV_jit(bev_map, points, grid_limits, grid_sizes)
        bev_map_pt = torch.from_numpy(bev_map)
        end_jit = time.time()

        print("Elapsed time: {}".format(end_jit - start_jit))

    def test_load(self):

        numpy_load_path = "/Users/erikbohnsack/data/training/np_bevs/0000"
        pytorch_load_path = "/Users/erikbohnsack/data/training/bevs/0000"
        dummy = 100
        start = time.time()
        for i in range(dummy):
            np_array = np.load(os.path.join(numpy_load_path, "000004.npy"))
            np_array1 = np.load(os.path.join(numpy_load_path, "000005.npy"))
            np_array2 = np.load(os.path.join(numpy_load_path, "000006.npy"))
        end = time.time()

        print("NP Loading: {}".format(end - start))
        print("Shapes: \n{}\n{}\n{}".format(np_array.shape, np_array1.shape, np_array2.shape))

        start = time.time()
        for i in range(dummy):
            t0 = torch.load(os.path.join(pytorch_load_path, "000004.pt"))
            t1 = torch.load(os.path.join(pytorch_load_path, "000005.pt"))
            t2 = torch.load(os.path.join(pytorch_load_path, "000006.pt"))
        end = time.time()

        print("Torch Loading: {}".format(end - start))
        print("Shapes: \n{}\n{}\n{}".format(t0.shape, t1.shape, t2.shape))


    def test_height_dimensions_density(self):
        config_path = 'cfg/cfg_mac.yml'
        config = load_config(config_path)
        input_config = InputConfig(config['INPUT_CONFIG'])
        basepath = "/Users/erikbohnsack/data/training"
        grid_limits = np.array([[input_config.x_min, input_config.x_max],
                                [input_config.y_min, input_config.y_max],
                                [input_config.z_min, input_config.z_max]])
        grid_sizes = np.array([input_config.x_grid_size, input_config.y_grid_size, input_config.z_grid_size])
        haffla = np.zeros(20,)
        for seq in range(20):
            for frame in range(10):
                file = os.path.join(basepath, 'velodyne', str(seq).zfill(4), str(frame).zfill(6) + '.bin')
                points = np.fromfile(file, dtype=np.float32).reshape((-1, 4))
                bev_map = np.zeros(input_config.bev_shape)

                kitti_dataset.format_to_BEV_jit(bev_map, points, grid_limits, grid_sizes)
                haffla += np.sum(bev_map, axis=(0, 1))
        print(haffla/np.linalg.norm(haffla, ord=1))
