import unittest
from fafe_utils.plot_stuff import plot_BEV
import pykitti
import time
import numpy as np
from fafe_utils import kitti_dataset
from cfg.config import InputConfig, TrainConfig
from cfg.config_stuff import load_config
from loss.box_utils import point_form_3d, rotate_3d_bbx
import torch


class TestPlot(unittest.TestCase):
    def test_plot_bevs(self):
        config_path = 'cfg_mini.yml'
        config = load_config(config_path)

        input_config = InputConfig(config['INPUT_CONFIG'])
        train_config = TrainConfig(config['TRAIN_CONFIG'])
        basepath = "/Users/erikbohnsack/data/training"

        tracking = pykitti.tracking(base_path=basepath, sequence="0000")
        points = tracking.get_velo(0)
        bev_map = np.zeros(input_config.bev_shape)
        print("BEV map size: {}".format(bev_map.shape))

        grid_limits = np.array([[input_config.x_min, input_config.x_max],
                                [input_config.y_min, input_config.y_max],
                                [input_config.z_min, input_config.z_max]])
        grid_sizes = np.array([input_config.x_grid_size, input_config.y_grid_size, input_config.z_grid_size])

        start_jit = time.time()
        kitti_dataset.format_to_BEV_jit(bev_map, points, grid_limits, grid_sizes)
        end_jit = time.time()

        plot_BEV(bev_map)
        print("Elapsed time: {}".format(end_jit - start_jit))

    def test_point_form(self):
        N = 10
        boxes = torch.rand((N, 4))
        z_center = 1
        z_height = 1.5

        angle = 1.57
        out = point_form_3d(boxes, z_center, z_height)
        print(out)

    def test_rotate_3d_bbox(self):
        N = 10
        boxes = torch.rand((N, 4))
        z_center = 1
        z_height = 1.5

        angle = 1.57
        angles = torch.rand((N, 1))
        out = point_form_3d(boxes, z_center, z_height)
        rotate_3d_bbx(out, angles)
