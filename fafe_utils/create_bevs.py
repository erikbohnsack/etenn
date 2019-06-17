import numpy as np
from tqdm import tqdm
from cfg.config import InputConfig
from fafe_utils import kitti_dataset
import torch
import pykitti
import os
import platform
from train import INPUT_CONFIG


def create_temporal_bevs(CONF):
    conf = InputConfig(CONF)
    if platform.system() == 'Darwin':
        basepath = "/Users/erikbohnsack/data/training"
    else:
        basepath = "/home/mlt/data/training"
    velodyne_path = os.path.join(basepath, 'velodyne')
    allfiles = os.listdir(velodyne_path)
    sequences = [fname for fname in allfiles if not fname.endswith('.DS_Store')]
    for sequence in sequences:
        tracking = pykitti.tracking(base_path=basepath, sequence=sequence)
        nof_frames = len(tracking.velo_files)

        bev_maps = []
        for frame in tqdm(range(0, nof_frames)):
            points = tracking.get_velo(frame)
            bev_map = torch.zeros(conf.bev_shape)
            kitti_dataset.format_to_BEV(bev_map, points, conf)

            bev_maps.append(bev_map)

        bev_path = os.path.join(os.path.join(basepath, 'bevs'), sequence)
        if not os.path.exists(bev_path):
            os.mkdir(bev_path)

        for i in range(conf.num_conseq_frames - 1, nof_frames - conf.num_conseq_frames + 1):
            conc_bev_map = torch.cat((bev_maps[i - conf.num_conseq_frames + 1:i + 1]), 2)
            assert np.shape(conc_bev_map)[2] == conf.num_conseq_frames * conf.bev_shape[2]
            # For saving/loading pytorch tensors:
            # torch.save(tensor, 'file.pt') and torch.load('file.pt')
            filename = os.path.join(bev_path, str(i).zfill(6) + '.pt')
            torch.save(conc_bev_map, filename)


def create_temporal_bevs_np_jit(CONF):
    conf = InputConfig(CONF)
    if platform.system() == 'Darwin':
        basepath = "/Users/erikbohnsack/data/training"
    else:
        basepath = "/home/mlt/data/training"
    velodyne_path = os.path.join(basepath, 'velodyne')
    allfiles = os.listdir(velodyne_path)
    sequences = [fname for fname in allfiles if not fname.endswith('.DS_Store')]
    grid_limits = np.array([[conf.x_min, conf.x_max],
                            [conf.y_min, conf.y_max],
                            [conf.z_min, conf.z_max]])
    grid_sizes = np.array([conf.x_grid_size, conf.y_grid_size, conf.z_grid_size])

    for sequence in sequences:
        tracking = pykitti.tracking(base_path=basepath, sequence=sequence)
        nof_frames = len(tracking.velo_files)

        bev_maps = []
        for frame in tqdm(range(0, nof_frames)):
            points = tracking.get_velo(frame)
            bev_map = np.zeros(conf.bev_shape)
            kitti_dataset.format_to_BEV_jit(bev_map, points, grid_limits, grid_sizes)
            print("Frame: {}, bev map shape {}".format(frame, bev_map.shape))
            bev_maps.append(bev_map)


        bev_path = os.path.join(os.path.join(basepath, 'bevs'), sequence)
        if not os.path.exists(bev_path):
            os.mkdir(bev_path)

        for i in range(conf.num_conseq_frames - 1, nof_frames - conf.num_conseq_frames + 1):
            conc_bev_map = np.concatenate((bev_maps[i - conf.num_conseq_frames + 1:i + 1]), 2)
            print("Conc bev map shape {}".format(conc_bev_map.shape))
            assert np.shape(conc_bev_map)[2] == conf.num_conseq_frames * conf.bev_shape[2]
            # For saving/loading pytorch tensors:
            # torch.save(tensor, 'file.pt') and torch.load('file.pt')
            filename = os.path.join(bev_path, str(i).zfill(6) + '.pt')
            torch.save(torch.from_numpy(conc_bev_map), filename)



if __name__ == "__main__":

    create_temporal_bevs_np_jit(INPUT_CONFIG)
