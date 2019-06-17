from cfg.config_stuff import load_config, save_config, get_root_dir, get_showroom_path
from models.fafepillar import PillarOfFafe
from models.fafenet import FafeNet
from models.little_fafe import LittleFafe
from fafe_utils.kitti_dataset import VoxelDataset
from loss.loss import *
from cfg.config import InputConfig, TrainConfig, LossConfig, ModelConfig
from torch.utils.data import DataLoader, ConcatDataset
import torch.optim as optim
import os
from time import time, strftime
from datetime import datetime, timedelta
import visdom
import fafe_utils.visdom_stuff as viz
from fafe_utils.plot_stuff import plot_grad_flow


def train():
    if os.path.exists('/home/mlt/mot/fafe/cfg/adams_computer'):
        config_path = 'cfg/cfg_pp_mini.yml'
    elif os.path.exists('/Users/erikbohnsack'):
        config_path = 'cfg/cfg_mac.yml'
    else:
        config_path = 'cfg/cfg_pp.yml'

    print('Using config: \n\t{}\n'.format(config_path))
    config = load_config(config_path)
    input_config = InputConfig(config["INPUT_CONFIG"])
    train_config = TrainConfig(config["TRAIN_CONFIG"])
    loss_config = LossConfig(config['LOSS_CONFIG'])
    model_config = ModelConfig(config['MODEL_CONFIG'])


    verbose = train_config.verbose

    time_str = strftime("%Y-%m-%d_%H-%M")
    weights_filename = 'trained_models/' + time_str + '/weights_' + time_str

    showroom_path = get_showroom_path(model_path="_".join(('weights', time_str)), full_path_bool=False)
    if not os.path.exists('trained_models/' + time_str):
        os.mkdir('trained_models/' + time_str)

    print('Training weights will be saved to:\n\t{}\n'.format(weights_filename))

    config_filename = 'trained_models/' + time_str + '/config_' + time_str + '.yml'
    save_config(config_filename, config)
    print('Config file saved to:\n\t{}\n'.format(config_filename))

    if train_config.use_visdom:
        print('Dont forget to run "visdom" in a terminal in parallel to this in order to start the Visdom server')
        print('Choose port with "python -m visdom.server -port {}" \n'.format(train_config.visdom_port))
        vis = visdom.Visdom(port=train_config.visdom_port)  # port 8097 is default
        loss_window, sub_loss_window, recall_window, precision_window = viz.get_windows(vis, time_str)
    print("~" * 20)
    print("Setting up net")
    pillar = PillarOfFafe(input_config=input_config, batch_size=train_config.batch_size,
                          verbose=input_config.pp_verbose)

    if model_config.model == 'little_fafe':
        fafe = LittleFafe(input_config=input_config)
    else:
        fafe = FafeNet(input_config=input_config)

    #########################
    # Set which device run on
    #########################
    if train_config.use_cuda:
        # "If you load your samples in the Dataset on CPU and would like to push it during training to the GPU,
        # you can speed up the host to device transfer by enabling pin_memory."
        # - ptrblck [https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723]
        pin_memory = False
        device = torch.device("cuda:" + str(train_config.cuda_device))
        print('\nUsing device {}\n'.format(device))
        pillar = pillar.to(device)
        fafe = fafe.to(device)
    else:
        pin_memory = False
        device = torch.device("cpu")
        print('Using CPU\n')

    loss_func = FafeLoss(input_config, train_config, loss_config, device)
    if train_config.use_cuda:
        loss_func = loss_func.to(device)

    print("Net set up successfully!")
    pp_pytorch_total_params = sum(p.numel() for p in pillar.parameters())
    pp_pytorch_trainable_params = sum(p.numel() for p in pillar.parameters() if p.requires_grad)
    print("~" * 20)
    print("PP:\n\tNumber of parameters: {}\n\tNumber of trainable parameters: {}".format(pp_pytorch_total_params,
                                                                                         pp_pytorch_trainable_params))

    pytorch_total_params = sum(p.numel() for p in fafe.parameters())
    pytorch_trainable_params = sum(p.numel() for p in fafe.parameters() if p.requires_grad)
    print("FAFE:\n\tNumber of parameters: {}\n\tNumber of trainable parameters: {}".format(pytorch_total_params,
                                                                                           pytorch_trainable_params))
    root_dir = get_root_dir()

    #########################
    # Define optimizer
    #########################
    params = list(pillar.parameters()) + list(fafe.parameters()) + list(loss_func.parameters())
    optimizer = optim.Adam(params,
                           lr=train_config.learning_rate,
                           weight_decay=train_config.weight_decay)
    print('Adams Optimizer set up with\n\tlr = {}\n\twd = {}\n'.format(train_config.learning_rate,
                                                                       train_config.weight_decay))

    #########################
    # Get Datasets
    #########################
    print('Training Data:')
    # fafe_sampler = FafeSampler(data_source=dataset, input_config=input_config)
    # TODO: Sampler needs a *data_source* as input to know the length of objects it can iterate over.
    # TODO: When concatenating

    training_dataloader = DataLoader(ConcatDataset([VoxelDataset(input_config, root_dir, split='training',
                                               sequence=seq) for seq in train_config.training_seqs]),
                                     batch_size=train_config.batch_size,
                                     shuffle=train_config.shuffle,
                                     num_workers=train_config.num_workers,
                                     pin_memory=pin_memory)
    print('Validation Data:')
    validation_dataloader = DataLoader(ConcatDataset([VoxelDataset(input_config, root_dir, split='training',
                                                                   sequence=seq) for seq in
                                                      train_config.validation_seqs]),
                                       batch_size=train_config.batch_size,
                                       shuffle=train_config.shuffle,
                                       num_workers=train_config.num_workers,
                                       pin_memory=pin_memory)
    print('Data Loaders set up with:\n\tBatch size:  {}\n\tNum Workers: {}'.format(train_config.batch_size,
                                                                                   train_config.num_workers))

    ###############################
    # Start training and evaluation
    ###############################
    print('\nTraining initiated [' + strftime("%Y-%m-%d %H:%M") + ']')
    for epoch in range(train_config.max_epochs):
        train_mean_loss, train_mean_recall, train_mean_precision, train_num_samples = 0, 0, 0, 0
        eval_mean_loss, eval_mean_recall, eval_mean_precision, eval_num_samples = 0, 0, 0, 0
        train_scaled_reg_mean, train_scaled_euler_mean, train_classification_loss = 0, 0, 0
        eval_scaled_reg_mean, eval_scaled_euler_mean, eval_classification_loss = 0, 0, 0

        #########################
        # TRAINING
        #########################
        tic = time()
        pillar.train()
        fafe.train()
        torch.set_grad_enabled(True)
        for i_batch, batch in enumerate(training_dataloader):

            # Always reset optimizer's gradient each iteration
            optimizer.zero_grad()

            # Create Pillar Pseudo Img
            voxel_stack, coord_stack, num_points_stack, num_nonempty_voxels, target, info = batch

            # Move all input data to the correct device if not using CPU
            if train_config.use_cuda:
                voxel_stack = voxel_stack.to(device)
                coord_stack = coord_stack.to(device)
                num_points_stack = num_points_stack.to(device)
                num_nonempty_voxels = num_nonempty_voxels.to(device)
                target = target.to(device)

            pseudo_stack = []
            for time_iter in range(input_config.num_conseq_frames):
                if input_config.pp_verbose:
                    print("~" * 20)
                    print("time iter: {}".format(time_iter))
                    print("voxels \n\tshape: {}".format(voxel_stack[:, time_iter].shape))
                    print("coord \n\tshape: {}".format(coord_stack[:, time_iter].shape))
                    print("num_points \n\tshape: {}".format(num_points_stack[:, time_iter].shape))
                    print("num_nonempty_voxels \n\tshape: {}".format(num_nonempty_voxels[:, time_iter].shape))

                pseudo_image = pillar(voxel_stack[:, time_iter], num_points_stack[:, time_iter],
                                      coord_stack[:, time_iter], num_nonempty_voxels[:, time_iter])

                if input_config.pp_verbose:
                    print("Pseudo_img: {}".format(pseudo_image.unsqueeze(1).shape))
                pseudo_stack.append(pseudo_image.unsqueeze(1))

            pseudo_torch = torch.cat(pseudo_stack, dim=1)
            if input_config.pp_verbose:
                print("Pseudo stacked over time: \n\t{}".format(pseudo_torch.shape))
            if train_config.use_cuda:
                pseudo_torch = pseudo_torch.to(device)
                target = target.to(device)

            # Forward propagation. Reshape by squishing together Time and Channel dimensions.
            # Reshaping basically as we do with BEV, stacking them on top of eachother.
            out_detection, out_regression = fafe.forward(
                pseudo_torch.reshape(-1, pseudo_torch.shape[1] * pseudo_torch.shape[2], pseudo_torch.shape[-2],
                                     pseudo_torch.shape[-1]))

            # Calculate the loss
            loss, recall, precision, scaled_reg, scaled_euler, classification_loss = loss_func(out_detection,
                                                                                              out_regression, target,
                                                                                              verbose)
            # Back propagate
            loss.backward()

            if train_config.plot_grad_flow:
                plot_grad_flow(pillar.named_parameters(),
                               os.path.join(showroom_path, 'grad_flow_pillar', "".join(
                                   ("epoch_", str(epoch).zfill(4), "_batch_", str(i_batch).zfill(4), ".png"))))
                plot_grad_flow(fafe.named_parameters(),
                               os.path.join(showroom_path, 'grad_flow_fafe', "".join(
                                   ("epoch_", str(epoch).zfill(4), "_batch_", str(i_batch).zfill(4), ".png"))))

            # Update the weights
            optimizer.step()

            train_mean_loss += loss
            train_mean_recall += recall
            train_mean_precision += precision
            train_scaled_reg_mean += scaled_reg
            train_scaled_euler_mean += scaled_euler
            train_classification_loss += classification_loss
            train_num_samples += 1

        # Calculate the actual averages
        train_mean_loss /= train_num_samples
        train_mean_recall /= train_num_samples
        train_mean_precision /= train_num_samples
        train_scaled_reg_mean /= train_num_samples
        train_scaled_euler_mean /= train_num_samples
        train_classification_loss /= train_num_samples
        training_time = time() - tic

        #########################
        # EVALUATION
        #########################
        tic2 = time()
        pillar.eval()
        fafe.eval()
        with torch.no_grad():
            for i_batch, batch in enumerate(validation_dataloader):
                # Create Pillar Pseudo Img
                voxel_stack, coord_stack, num_points_stack, num_nonempty_voxels, target, index = batch

                # Move all input data to the correct device if not using CPU
                if train_config.use_cuda:
                    voxel_stack = voxel_stack.to(device)
                    coord_stack = coord_stack.to(device)
                    num_points_stack = num_points_stack.to(device)
                    num_nonempty_voxels = num_nonempty_voxels.to(device)
                    target = target.to(device)

                pseudo_stack = []
                for time_iter in range(input_config.num_conseq_frames):
                    if train_config.verbose:
                        print("~" * 20)
                        print("time iter: {}".format(time_iter))
                        print("voxels \n\tshape: {}".format(voxel_stack[:, time_iter].shape))
                        print("coord \n\tshape: {}".format(coord_stack[:, time_iter].shape))
                        print("num_points \n\tshape: {}".format(num_points_stack[:, time_iter].shape))
                        print("num_nonempty_voxels \n\tshape: {}".format(num_nonempty_voxels[:, time_iter].shape))

                    pseudo_image = pillar(voxel_stack[:, time_iter], num_points_stack[:, time_iter],
                                          coord_stack[:, time_iter], num_nonempty_voxels[:, time_iter])

                    if train_config.verbose:
                        print("Pseudo_img: {}".format(pseudo_image.unsqueeze(1).shape))
                    pseudo_stack.append(pseudo_image.unsqueeze(1))

                pseudo_torch = torch.cat(pseudo_stack, dim=1)
                if train_config.use_cuda:
                    pseudo_torch = pseudo_torch.to(device)
                    target = target.to(device)

                # Forward propagation
                out_detection, out_regression = fafe.forward(
                    pseudo_torch.reshape(-1, pseudo_torch.shape[1] * pseudo_torch.shape[2], pseudo_torch.shape[-2],
                                         pseudo_torch.shape[-1]))
                # Calculate the loss
                loss, recall, precision, scaled_reg, scaled_euler, classification_loss = loss_func(out_detection,
                                                                                                  out_regression,
                                                                                                  target, verbose)
                eval_mean_loss += loss
                eval_mean_recall += recall
                eval_mean_precision += precision
                eval_scaled_reg_mean += scaled_reg
                eval_scaled_euler_mean += scaled_euler
                eval_classification_loss += classification_loss
                eval_num_samples += 1
        eval_mean_loss /= eval_num_samples
        eval_mean_recall /= eval_num_samples
        eval_mean_precision /= eval_num_samples
        eval_scaled_reg_mean /= eval_num_samples
        eval_scaled_euler_mean /= eval_num_samples
        eval_classification_loss /= eval_num_samples

        eval_time = time() - tic2
        total_time = time() - tic

        #########################
        # PRINT STUFF ON SCREEN
        #########################
        print('\nEpoch {} / {}\n{}\nCurrent time: {}'.format(epoch,
                                                             train_config.max_epochs - 1,
                                                             '-' * 12,
                                                             strftime("%Y-%m-%d %H:%M")))
        print('Epoch Total Time: {} s ({} + {})'.format(round(total_time, 2),
                                                        round(training_time, 2),
                                                        round(eval_time, 2)))
        print('Next Epoch ETA: ' + format(datetime.now() + timedelta(seconds=total_time), '%Y-%m-%d %H:%M'))
        print('Training ETA: ' + format(
            datetime.now() + timedelta(seconds=total_time * (train_config.max_epochs - epoch - 1)), '%Y-%m-%d %H:%M'))
        print('Train\n\tLoss: \t\t{}'
              '\n\t\tL1:  \t{}'
              '\n\t\tEuler:\t{}'
              '\n\t\tCL:  \t{}'
              '\n\tRecall: \t{}'
              '\n\tPrecision:\t{}'.format(train_mean_loss,
                                          train_scaled_reg_mean,
                                          train_scaled_euler_mean,
                                          train_classification_loss,
                                          train_mean_recall,
                                          train_mean_precision))
        print('Validation\n\tLoss: \t\t{}'
              '\n\t\tL1:  \t{}'
              '\n\t\tEuler:\t{}'
              '\n\t\tCL:  \t{}'
              '\n\tRecall: \t{}'
              '\n\tPrecision:\t{}'.format(eval_mean_loss,
                                          eval_scaled_reg_mean,
                                          eval_scaled_euler_mean,
                                          eval_classification_loss,
                                          eval_mean_recall,
                                          eval_mean_precision))
        if train_config.use_visdom:
            # Visualize Loss
            viz.push_data(epoch, vis,
                          loss_window, sub_loss_window, recall_window, precision_window,
                          train_mean_loss, eval_mean_loss,
                          train_scaled_reg_mean, train_classification_loss,
                          train_scaled_euler_mean, eval_scaled_euler_mean,
                          eval_scaled_reg_mean, eval_classification_loss,
                          train_mean_recall, eval_mean_recall,
                          train_mean_precision, eval_mean_precision)

        ####################################################
        # SAVE WEIGHTS (every save_weights_modulus th epoch)
        ####################################################
        if epoch % train_config.save_weights_modulus == 0 or epoch == train_config.max_epochs - 1:
            save_filename = weights_filename + '_epoch_' + str(epoch)

            pp_fn = save_filename + '_pp'
            torch.save({
                'epoch': epoch,
                'model_state_dict': pillar.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss},
                pp_fn)

            fafe_fn = save_filename + '_fafe'
            torch.save({
                'epoch': epoch,
                'model_state_dict': fafe.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss},
                fafe_fn)

    print('Training Complete [' + strftime("%Y-%m-%d %H:%M") + ']')


if __name__ == "__main__":
    train()
