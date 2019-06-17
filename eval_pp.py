import torch
import numpy as np
from models.fafenet import FafeNet
from models.little_fafe import LittleFafe
from models.fafepillar import PillarOfFafe
from models.fafepredict import FafePredict
from fafe_utils.kitti_dataset import VoxelDataset
from torch.utils.data import DataLoader, ConcatDataset
from cfg.config import InputConfig, TrainConfig, EvalConfig, ModelConfig, PostConfig
from loss.box_utils import point_form_fafe, point_form_3d, center_size_3d, rotate_3d_bbx, translate_center
from fafe_utils.plot_stuff import draw_lidar, draw_gt_boxes3d, draw_points_3d, draw_gospa
from fafe_utils.imu import load_imu
import os
import mayavi
from time import strftime, time
from cfg.config_stuff import load_config, get_showroom_path, get_root_dir
from fafe_utils.eval_metrics import GOSPA
from fafe_utils.fafe_utils import within_fov, too_close


def eval_with_GT(fafe_model_path, pp_model_path, data_path, config_path):
    timestr = strftime("%Y-%m-%d_%H:%M")

    showroom_path = get_showroom_path(fafe_model_path, full_path_bool=True)
    print('Images will be saved to:\n\t{}'.format(showroom_path))

    velo_path = os.path.join(data_path, 'training', 'velodyne')

    config = load_config(config_path)
    start_time = time()
    input_config = InputConfig(config['INPUT_CONFIG'])
    train_config = TrainConfig(config['TRAIN_CONFIG'])
    eval_config = EvalConfig(config['EVAL_CONFIG'])
    model_config = ModelConfig(config['MODEL_CONFIG'])
    post_config = PostConfig(config['POST_CONFIG'])

    pillar = PillarOfFafe(input_config, train_config.batch_size, train_config.verbose)

    if model_config.model == 'little_fafe':
        fafe = LittleFafe(input_config)
    else:
        fafe = FafeNet(input_config)

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
        pillar = pillar.to(device)
        fafe = fafe.to(device)
    else:
        device = torch.device("cpu")
        print('Using CPU\n')

    imu_path = os.path.join(data_path, 'training', 'oxts')

    pillar.load_state_dict(torch.load(pp_model_path, map_location=lambda storage, loc: storage)["model_state_dict"])
    fafe.load_state_dict(torch.load(fafe_model_path, map_location=lambda storage, loc: storage)["model_state_dict"])
    pillar.eval()
    fafe.eval()

    print("Model loaded, loading time: {}".format(round(time() - start_time, 2)))
    timestr = strftime("%Y-%m-%d_%H:%M")
    eval_head = FafePredict(input_config, eval_config)
    eval_head.eval()
    print("Evalutation Model Loaded!")

    testing_datasets = [VoxelDataset(input_config, root_dir=data_path, split='training', sequence=seq) for seq in
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

        # Create Pillar Pseudo Img
        voxel_stack, coord_stack, num_points_stack, num_nonempty_voxels, target, info = batch

        # Move all input data to the correct device if not using CPU
        if train_config.use_cuda:
            voxel_stack = voxel_stack.to(device)
            coord_stack = coord_stack.to(device)
            num_points_stack = num_points_stack.to(device)
            num_nonempty_voxels = num_nonempty_voxels.to(device)
            target = target.to(device)

        timeit = time()
        with torch.no_grad():
            pseudo_stack = []
            for time_iter in range(input_config.num_conseq_frames):
                pseudo_image = pillar(voxel_stack[:, time_iter], num_points_stack[:, time_iter],
                                      coord_stack[:, time_iter], num_nonempty_voxels[:, time_iter])

                if input_config.pp_verbose:
                    print("Pseudo_img: {}".format(pseudo_image.unsqueeze(1).shape))
                pseudo_stack.append(pseudo_image.unsqueeze(1))

            pseudo_torch = torch.cat(pseudo_stack, dim=1)

            if train_config.use_cuda:
                pseudo_torch = pseudo_torch.to(device)
                target = target.to(device)

            # Forward propagation. Reshape by squishing together Time and Channel dimensions.
            # Reshaping basically as we do with BEV, stacking them on top of eachother.
            out_detection, out_regression = fafe.forward(
                pseudo_torch.reshape(-1, pseudo_torch.shape[1] * pseudo_torch.shape[2], pseudo_torch.shape[-2],
                                     pseudo_torch.shape[-1]))
            inference = eval_head(out_detection, out_regression)
        print("\nTime to propagate through network: {}".format(round(time() - timeit, 2)))

        nB = target.shape[0]
        nT = input_config.num_conseq_frames

        tic = time()
        for batch_index in range(nB):
            current_sequence = info["sequence"][batch_index]
            frame = info["GT_indices"][0]

            if torch.is_tensor(current_sequence):
                current_sequence = current_sequence.item()

            imud = load_imu(imu_path, current_sequence)
            print("Batch: {} \t| Seq: {} Frame: {}".format(batch_index, current_sequence, frame.item()), end='')

            if eval_config.save_raw:
                if not os.path.exists("showroom/detections_" +str(current_sequence).zfill(4)):
                    os.mkdir("showroom/detections_" +str(current_sequence).zfill(4))
                filename = os.path.join("showroom", "detections_{}".format(str(current_sequence).zfill(4)), str(int(frame)).zfill(4) + ".pt")
                torch.save(inference, filename)

            if eval_config.save_figs:
                file = os.path.join(velo_path, str(current_sequence).zfill(4), str(frame.item()).zfill(6) + '.bin')
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
                point_form_gt_3d = point_form_3d(point_form_gt, input_config.z_center, input_config.z_height,
                                                 device=device)

                # Rotate gt 3D bboxes
                gt_center = center_size_3d(non_zero_gt, input_config.z_center, device)
                point_form_gt_3d_rot = rotate_3d_bbx(point_form_gt_3d, non_zero_gt[:, 4], gt_center, device)

                # Get inference
                inference_reg = inference[batch_index][time_index]

                # Filter out if outside FOV
                filtered_inference_reg = []
                for inf_reg in inference_reg:
                    np_inf_reg = inf_reg.cpu().data.numpy()
                    if within_fov(np_inf_reg[0:2], min_angle=0.78, max_angle=2.45, max_radius=100):
                        filtered_inference_reg.append(np_inf_reg)

                # Filter out if too closely located
                filtered_inference_reg = too_close(filtered_inference_reg, eval_config.distance_threshold)

                if eval_config.save_detections_as_measurements and time_index == 0:
                    for fir in filtered_inference_reg:
                        file1 = open(showroom_path + "/detections_" + str(current_sequence).zfill(4) + ".txt", "a")
                        # Save it on the format PMBM wants it... p = [x, y, rot] with x defined right wards and y forward
                        file1.write(str(frame.item()) + ' Car:' + str(fir[5]) + ' ' + str(- fir[1]) + ' ' + str(
                            fir[0]) + ' ' + str(fir[4]) + "\n")
                        file1.close()

                # Make filtered_inference_reg numpy because GOSPA needs it to be
                filtered_inference_reg = np.array(filtered_inference_reg)

                # Calculate GOSPA if ordered to do so
                if eval_config.use_gospa:
                    if inference_reg.is_cuda:
                        gospa_gt = non_zero_gt[:, 0:2].cpu()
                        # gospa_inf = inference_reg[:, 0:2].cpu()
                        gospa_inf = filtered_inference_reg[:, 0:2] if len(filtered_inference_reg) > 0 else np.array([])
                    else:
                        gospa_gt = non_zero_gt[:, 0:2]
                        # gospa_inf = inference_reg[:, 0:2]
                        gospa_inf = filtered_inference_reg[:, 0:2]

                    gospa_score = GOSPA(gospa_gt, gospa_inf)
                    gospa_scores_dict[int(current_sequence), current_frame][time_index] = gospa_score
                    if eval_config.save_figs and eval_config.draw_gospa:
                        fig = draw_gospa(gospa_score, fig, time_index)

                # Make filtered_inference_reg torch tensor because point forms need it to be
                filtered_inference_reg = torch.from_numpy(filtered_inference_reg).to(device)

                # If no outputs -> plot ev. gt and continue
                if len(filtered_inference_reg) == 0:
                    if time_index == 0:
                        fig = draw_gt_boxes3d(point_form_gt_3d_rot, fig, color=eval_config.gt_color)
                        continue
                    else:
                        # Translate center points according to time_index
                        translated_gt_c = translate_center(gt_center, imu_data=imud[frame.item()], timestep=time_index,
                                                           dt=input_config.dt, device=device)
                        fig = draw_points_3d(translated_gt_c, fig, color=eval_config.gt_color)
                        continue

                # Get probabilities of objects
                inference_probabilities = filtered_inference_reg[..., -1] if eval_config.show_confidence else None

                # Transform inference to 2D bbox x1y1x2y2
                inf_points_2d = point_form_fafe(filtered_inference_reg)

                # Transform inference to 3D bbox
                inference_points = point_form_3d(inf_points_2d, input_config.z_center, input_config.z_height, device)

                # Rotate inference 3D bboxes
                inference_center = center_size_3d(filtered_inference_reg, input_config.z_center, device)
                inference_points_rot = rotate_3d_bbx(inference_points, filtered_inference_reg[:, 4], inference_center,
                                                     device)

                # Translate center points according to time_index
                translated_gt_c = translate_center(gt_center, imu_data=imud[frame.item()], timestep=time_index,
                                                   dt=input_config.dt, device=device)
                translated_infer_c = translate_center(inference_center, imu_data=imud[frame.item()],
                                                      timestep=time_index, dt=input_config.dt, device=device)

                # TODO: Move these if-statements further up to reduce computation
                if eval_config.save_figs:
                    if time_index == 0:
                        fig = draw_gt_boxes3d(point_form_gt_3d_rot, fig, color=eval_config.gt_color)
                        fig = draw_gt_boxes3d(inference_points_rot, fig, color=eval_config.infer_color,
                                              probabilities=inference_probabilities)
                    else:
                        fig = draw_points_3d(translated_gt_c, fig, color=eval_config.gt_color)
                        fig = draw_points_3d(translated_infer_c, fig, color=eval_config.infer_color)
            if eval_config.save_figs:
                filename = "_".join(
                    ("Infer", "seq", str(current_sequence), "f", str(frame.item()), timestr + ".png"))
                filepath = os.path.join(showroom_path, 'oracle_view', filename)
                filename_top = "_".join(
                    ("Top_Infer", "seq", str(current_sequence), "f", str(frame.item()), timestr + ".png"))
                filepath_top = os.path.join(showroom_path, 'top_view', filename_top)
                mayavi.mlab.view(azimuth=90, elevation=0, focalpoint=[24., 0, 0],
                                 distance=120.0, figure=fig)
                mayavi.mlab.savefig(filepath_top, figure=fig)
                mayavi.mlab.view(azimuth=180, elevation=70, focalpoint=[12.0909996, -1.04700089, -2.03249991],
                                 distance=62.0, figure=fig)
                mayavi.mlab.savefig(filepath, figure=fig)

            if eval_config.use_gospa:
            # Save gospa scores in txt file
                file2 = open(showroom_path + "/gospa_scores_" + str(current_sequence).zfill(4) + ".txt", "a")
                row_str = str(frame.item())
                for time_index in range(nT):
                    row_str += (' ' + str(gospa_scores_dict[int(current_sequence), current_frame][time_index]))
                file2.write(row_str + "\n")
                file2.close()

            print('\t | time: {}'.format(round(time() - tic, 2)))


if __name__ == "__main__":
    if os.path.exists('/Users/erikbohnsack/Code/MOT/fafe/trained_models/weights_2019-04-29_16-34_epoch_40_pp'):
        pp_model_path = '/Users/erikbohnsack/Code/MOT/fafe/trained_models/weights_2019-04-29_16-34_epoch_40_pp'
        fafe_model_path = '/Users/erikbohnsack/Code/MOT/fafe/trained_models/weights_2019-04-29_16-34_epoch_40_fafe'
        config_path = 'cfg/cfg_mac.yml'
    elif os.path.exists('/home/mlt/mot/fafe/cfg/adams_computer'):
        root = '/home/mlt/mot/fafe/trained_models'

        if False:
            trained_model = '2019-05-02_12:32'
            fafe_weight = 'weights_2019-05-02_12-32_epoch_15_fafe'
            pp_weight = 'weights_2019-05-02_12-32_epoch_15_pp'
            cfg_name = 'config_2019-05-02_12-32.yml'

            fafe_model_path = os.path.join(root, trained_model, fafe_weight)
            pp_model_path = os.path.join(root, trained_model, pp_weight)
            config_path = os.path.join(root, trained_model, cfg_name)
        else:
            fafe_model_path = '/home/mlt/mot/fafe/trained_models/2019-05-03_14-01/weights_2019-05-03_14-01_epoch_110_fafe'
            pp_model_path = '/home/mlt/mot/fafe/trained_models/2019-05-03_14-01/weights_2019-05-03_14-01_epoch_110_pp'
            config_path = '/home/mlt/mot/fafe/trained_models/2019-05-03_14-01/config_2019-05-03_14-01.yml'
    else:
        fafe_model_path = '/home/mlt/mot/fafe/trained_models/2019-04-15_14:59/weights_2019-04-29_16-34_epoch_40_fafe'
        pp_model_path = '/home/mlt/mot/fafe/trained_models/2019-04-15_14:59/weights_2019-04-29_16-34_epoch_40_pp'
        config_path = '/home/mlt/mot/fafe/trained_models/config_2019-04-29_16-34.yml'
    data_path = get_root_dir()
    eval_with_GT(fafe_model_path=fafe_model_path, pp_model_path=pp_model_path, data_path=data_path,
                 config_path=config_path)
