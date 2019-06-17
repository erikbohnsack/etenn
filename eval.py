import torch
import numpy as np
from models.fafenet import FafeNet
from models.haveFafeNet import HaveFafe4Eval
from models.fafepredict import FafePredict
from fafe_utils.kitti_dataset import TestDataset, TemporalBEVsDataset
from torch.utils.data import DataLoader, ConcatDataset
from cfg.config import InputConfig, EvalConfig
from loss.box_utils import point_form_fafe, point_form_3d, center_size_3d, rotate_3d_bbx, translate_center
from fafe_utils.plot_stuff import draw_lidar, draw_gt_boxes3d, draw_points_3d, draw_gospa
from fafe_utils.imu import load_imu
import os
import mayavi
from time import strftime, time
from cfg.config_stuff import load_config, get_showroom_path, get_root_dir
from fafe_utils.eval_metrics import GOSPA

def eval(model_path, data_path):
    config_path = 'cfg/cfg_mini.yml'
    config = load_config(config_path)

    input_config = InputConfig(config['INPUT_CONFIG'])
    eval_config = EvalConfig(config['EVAL_CONFIG'])
    model = FafeNet(input_config)

    model.load_state_dict(torch.load(model_path)["model_state_dict"])
    model.eval()
    print("Model loaded")
    eval_head = FafePredict(input_config, eval_config)
    testing_seqs = [0]  # TODO: Fix

    testing_datasets = [TestDataset(input_config, root_dir=data_path, split='testing', sequence=seq) for seq in
                        testing_seqs]

    testing_dataset = ConcatDataset(testing_datasets)
    testing_loader = DataLoader(dataset=testing_dataset, batch_size=eval_config.batch_size,
                                num_workers=eval_config.num_workers)

    for batch_index, batch in enumerate(testing_loader):
        input, info = batch
        out_det, out_reg = model(input)
        inference = eval_head(out_det, out_reg)


def eval_with_GT(model_path, data_path, config_path, v2=False):
    timestr = strftime("%Y-%m-%d_%H:%M")

    showroom_path = get_showroom_path(model_path, full_path_bool=True)
    print('Images will be saved to:\n\t{}'.format(showroom_path))

    velo_path = os.path.join(data_path, 'training', 'velodyne')

    config = load_config(config_path)
    start_time = time()
    input_config = InputConfig(config['INPUT_CONFIG'])
    eval_config = EvalConfig(config['EVAL_CONFIG'])

    if v2:
        model = HaveFafe4Eval(input_config)
    else:
        model = FafeNet(input_config)

    if eval_config.use_cuda:
        if eval_config.cuda_device == 0:
            device = torch.device("cuda:0")
            print('Using CUDA:{}\n'.format(0))
        elif eval_config.cuda_device == 1:
            device = torch.device("cuda:1")
            print('Using CUDA:{}\n'.format(1))
        else:
            print('Functionality for CUDA device cuda:{} not yet implemented.'.format(eval_config.cuda_device))
            print('Using cuda:0 instead...\n')
            device = torch.device("cuda:0")
        model = model.to(device)
    else:
        device = torch.device("cpu")
        print('Using CPU\n')

    imu_path = os.path.join(data_path, 'training', 'oxts')

    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage)["model_state_dict"])
    model.eval()

    print("Model loaded, loading time: {}".format(round(time() - start_time,2)))
    timestr = strftime("%Y-%m-%d_%H:%M")
    eval_head = FafePredict(input_config, eval_config)
    eval_head.eval()
    print("Evalutation Model Loaded!")

    testing_datasets = [TemporalBEVsDataset(input_config, root_dir=data_path, split='training', sequence=seq) for seq in
                        eval_config.validation_seqs]

    testing_dataset = ConcatDataset(testing_datasets)

    testing_loader = DataLoader(dataset=testing_dataset,
                                batch_size=eval_config.batch_size,
                                num_workers=eval_config.num_workers,
                                shuffle=False)

    print("Starting to iterate over batches")

    if eval_config.use_gospa:
        gospa_scores_dict = {}

    for batch_idx, batch in enumerate(testing_loader):
        input, target, info = batch

        if eval_config.use_cuda:
            input = input.to(device)
            target = target.to(device)

        timeit = time()
        with torch.no_grad():
            out_det, out_reg = model(input)
            inference = eval_head(out_det, out_reg)
        print("\nTime to propagate through network: {}".format(round(time() - timeit,2)))

        nB = target.shape[0]
        nT = input_config.num_conseq_frames

        tic = time()
        for batch_index in range(nB):
            current_sequence = info["sequence"][batch_index]
            imud = load_imu(imu_path, current_sequence)
            print("Batch: {} \t| Seq: {}".format(batch_index, info["sequence"][batch_index]), end='')

            if eval_config.save_figs:
                frame = info["GT_indices"][0]
                file = os.path.join(velo_path, info["sequence"][batch_index], str(frame.item()).zfill(6) + '.bin')
                pc = np.fromfile(file, dtype=np.float32).reshape((-1, 4))
                fig = draw_lidar(pc, off_screen_rendering=eval_config.off_screen_rendering)

            if eval_config.use_gospa:
                current_frame = info["Current_index"][batch_index].item()
                gospa_scores_dict[int(current_sequence), current_frame] = np.zeros(nT)

            for time_index in range(nT):
                # Get ground truths
                non_zero_gt_mask = target[batch_index][time_index].sum(dim=1) != 0
                non_zero_gt = target[batch_index][time_index][non_zero_gt_mask]

                # Transform gt to 2D bbox x1y1x2y2 form
                point_form_gt = point_form_fafe(non_zero_gt)

                # Transform gt to 3D bbox form
                point_form_gt_3d = point_form_3d(point_form_gt, input_config.z_center, input_config.z_height, device=device)

                # Rotate gt 3D bboxes
                gt_center = center_size_3d(non_zero_gt, input_config.z_center, device)
                point_form_gt_3d_rot = rotate_3d_bbx(point_form_gt_3d, non_zero_gt[:, 4], gt_center, device)

                # Get inference
                inference_reg = inference[batch_index][time_index]

                # Calculate GOSPA if ordered to do so
                if eval_config.use_gospa:
                    if inference_reg.is_cuda:
                        gospa_gt = non_zero_gt[:,0:2].cpu()
                        gospa_inf = inference_reg[:,0:2].cpu()
                    else:
                        gospa_gt = non_zero_gt[:,0:2]
                        gospa_inf = inference_reg[:,0:2]
                    gospa_score = GOSPA(gospa_gt, gospa_inf)
                    gospa_scores_dict[int(current_sequence), current_frame][time_index] = gospa_score
                    if eval_config.save_figs and eval_config.draw_gospa:
                        fig = draw_gospa(gospa_score, fig, time_index)

                                   # Get probabilities of objects
                inference_probabilities = inference_reg[..., -1] if eval_config.show_confidence else None

                # Transform inference to 2D bbox x1y1x2y2
                inf_points_2d = point_form_fafe(inference_reg)

                # Transform inference to 3D bbox
                inference_points = point_form_3d(inf_points_2d, input_config.z_center, input_config.z_height, device)

                # Rotate inference 3D bboxes
                inference_center = center_size_3d(inference_reg, input_config.z_center, device)
                inference_points_rot = rotate_3d_bbx(inference_points, inference_reg[:, 4], inference_center, device)

                # Translate center points according to time_index
                translated_gt_c = translate_center(gt_center, imu_data=imud[frame.item()], timestep=time_index, dt=input_config.dt, device=device)
                translated_infer_c = translate_center(inference_center, imu_data=imud[frame.item()], timestep=time_index, dt=input_config.dt, device=device)


                # TODO: Move these if-statements further up to reduce computation
                if eval_config.save_figs:
                    if time_index == 0:
                        fig = draw_gt_boxes3d(point_form_gt_3d_rot, fig, color=eval_config.gt_color)
                        fig = draw_gt_boxes3d(inference_points_rot, fig, color=eval_config.infer_color, probabilities=inference_probabilities)
                    else:
                        fig = draw_points_3d(translated_gt_c, fig, color=eval_config.gt_color)
                        fig = draw_points_3d(translated_infer_c, fig, color=eval_config.infer_color)
            if eval_config.save_figs:
                filename = "_".join(("Infer", "seq", info["sequence"][batch_index], "f", str(frame.item()), timestr + ".png"))
                filepath = os.path.join(showroom_path, 'oracle_view', filename)
                filename_top = "_".join(("Top_Infer", "seq", info["sequence"][batch_index], "f", str(frame.item()), timestr + ".png"))
                filepath_top = os.path.join(showroom_path, 'top_view', filename_top)
                mayavi.mlab.view(azimuth=90, elevation=0, focalpoint=[24., 0, 0],
                                 distance=120.0, figure=fig)
                mayavi.mlab.savefig(filepath_top, figure=fig)
                mayavi.mlab.view(azimuth=180, elevation=70, focalpoint=[12.0909996, -1.04700089, -2.03249991],
                                 distance=62.0, figure=fig)
                mayavi.mlab.savefig(filepath, figure=fig)
            print('\t | time: {}'.format(round(time()-tic,2)))

        print(gospa_scores_dict)

if __name__ == "__main__":
    if os.path.exists('/Users/erikbohnsack/Code/MOT/fafe/trained_models/weights_2019-04-16_17_18_epoch_25'):
        model_path = '/Users/erikbohnsack/Code/MOT/fafe/trained_models/weights_2019-04-16_17_18_epoch_25'
        config_path = 'cfg/cfg_mac.yml'
    elif os.path.exists('/home/mlt/adam/fafe/trained_models/2019-04-15_14:59'):
        model_path = '/home/mlt/adam/fafe/trained_models/2019-04-15_14:59/weights_2019-04-15_14:59_epoch_50'
        config_path = 'cfg/cfg.yml'
    else:
        model_path = '/home/mlt/mot/fafe/trained_models/weights_2019-04-16_17_18_epoch_25'
        config_path = '/home/mlt/mot/fafe/trained_models/config_2019-04-16_17_18.yml'
    data_path = get_root_dir()
    eval_with_GT(model_path=model_path, data_path=data_path, config_path=config_path, v2=True)
