import torch
import numpy as np
from models.fafenet import FafeNet
from post_fafe import PostFafe
from models.little_fafe import LittleFafe
from models.fafepillar import PillarOfFafe
from models.fafepredict import FafePredict
from fafe_utils.kitti_dataset import TemporalBEVsDataset
from torch.utils.data import DataLoader
from cfg.config import InputConfig, TrainConfig, EvalConfig, ModelConfig, PostConfig
import os
from time import strftime, time
from cfg.config_stuff import load_config, get_showroom_path
from fafe_utils.post_fafe_utils import SingleTargetHypothesis, transf, translate_center
from fafe_utils.imu import load_imu


def eval_post_fafe(fafe_model_path, data_path, config_path, sequence, kitti):
    """

    :param self:
    :param fafe_model_path:
    :param pp_model_path:
    :param data_path:
    :param config_path:
    :return:
    """
    timestr = strftime("%Y-%m-%d_%H:%M")

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
    fafe.load_state_dict(torch.load(fafe_model_path, map_location=lambda storage, loc: storage)["model_state_dict"])
    post_fafe = PostFafe(input_config, post_config, device)

    pillar.eval()
    fafe.eval()

    print("Model loaded, loading time: {}".format(round(time() - start_time, 2)))
    timestr = strftime("%Y-%m-%d_%H:%M")
    eval_head = FafePredict(input_config, eval_config)
    eval_head.eval()
    print("Evalutation Model Loaded!")

    # for seq in eval_config.validation_seqs:
    testing_dataset = TemporalBEVsDataset(input_config, root_dir=data_path, split='training', sequence=sequence)
    testing_loader = DataLoader(dataset=testing_dataset,
                                batch_size=eval_config.batch_size,
                                num_workers=eval_config.num_workers,
                                shuffle=False)

    data = []
    frames = []
    print("Starting to run over sequence {}. Total number of frames: {}".format(sequence, len(testing_loader)))
    total_time_per_iteration = []
    for batch_idx, batch in enumerate(testing_loader):

        input, target, info = batch

        if eval_config.use_cuda:
            input = input.to(device)
            target = target.to(device)

        timeit = time()
        with torch.no_grad():
            out_det, out_reg = fafe(input)
            inference = eval_head(out_det, out_reg)
        inference_time = round(time() - timeit, 2)
        nB = target.shape[0]
        nT = input_config.num_conseq_frames
        current_sequence = sequence
        frame = info["GT_indices"][0]
        if type(frame) == torch.Tensor:
            frame = frame.item()
        d = {}
        d['current_time'] = frame
        d['frame_id'] = frame       # To match pmbm's data structure...

        frames.append(frame)
        imud = load_imu(imu_path, current_sequence)

        true_states = kitti.get_bev_states(frame, classes_to_track=['Car', 'Van'])
        d['true_states'] = true_states
        d['state_dims'] = 3

        if torch.is_tensor(current_sequence):
            current_sequence = current_sequence.item()

        meas_ = inference[0][0]
        meas = []
        for row in range(meas_.shape[0]):
            meas.append(transf(meas_[0].to('cpu').numpy()))
        d['measurements'] = meas
        if len(inference) > 1:
            raise ValueError("Batch index > 1. Do something god damnit")

        # Call post_fafe to infer the inference
        post_fafe(inference[0])
        imu_data = imud[frame]
        # Add targets and predictions to data dict
        # TID = Target ID
        estimated_targets = []
        for tid, tensor_state in post_fafe.object_state.items():
            continue_bool = 0
            if post_config.verbose:
                print("Object: {} \n\t{}".format(tid, tensor_state))
            _temp = {}
            _temp['target_idx'] = tid
            _temp['object_class'] = 'car'

            _state = tensor_state[0].numpy()

            single_target = SingleTargetHypothesis(transf(_state))
            _temp['single_target'] = single_target
            _temp['state_predictions'] = []
            _temp['var_predictions'] = []
            for i, step in enumerate(tensor_state):
                if step is None or continue_bool:
                    continue_bool = 1
                    continue

                # Coordinate transform predictions
                preds = transf(step.numpy())

                if post_config.coordinate_transform:
                    preds = translate_center(preds, imu_data, timestep=i, dt=input_config.dt )

                _temp['state_predictions'].append(preds)
                if post_config.verbose:
                    print("Step: {}".format(transf(step.numpy())))
            estimated_targets.append(_temp)

        d['estimated_targets'] = estimated_targets
        data.append(d)
        total_time_per_iteration.append(inference_time)
        print("{}({}s,{}#), ".format(frame, inference_time, len(estimated_targets)),end='')

    return data, total_time_per_iteration
