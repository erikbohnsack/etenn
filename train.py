from models.fafenet import FafeNet
from models.little_fafe import LittleFafe
from loss.loss import *
from fafe_utils.kitti_dataset import TemporalBEVsDataset
from cfg.config import InputConfig, TrainConfig, LossConfig, ModelConfig
from torch.utils.data import DataLoader, ConcatDataset
import torch.optim as optim
import os
from time import time, strftime
from datetime import datetime, timedelta
import visdom
import fafe_utils.visdom_stuff as viz
from cfg.config_stuff import load_config, save_config, get_root_dir


def train():
    if os.path.exists('/home/mlt/mot/fafe/cfg/adams_computer'):
        config_path = 'cfg/cfg_mini.yml'
    else:
        config_path = 'cfg/cfg.yml'

    print('Using config: \n\t{}\n'.format(config_path))
    config = load_config(config_path)

    input_config = InputConfig(config['INPUT_CONFIG'])
    train_config = TrainConfig(config['TRAIN_CONFIG'])
    loss_config = LossConfig(config['LOSS_CONFIG'])
    model_config = ModelConfig(config['MODEL_CONFIG'])

    verbose = train_config.verbose

    time_str = strftime("%Y-%m-%d_%H-%M")
    weights_filename = 'trained_models/' + time_str + '/weights_' + time_str

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

    # Get root directory depending on which computer is running...
    # Don't forget to add your own path in 'fafe_utils.config_stuff.get_root_dir'
    root_dir = get_root_dir()

    #########################
    # Define network
    #########################
    if model_config.model == 'little_fafe':
        net = LittleFafe(input_config=input_config)
    else:
        net = FafeNet(input_config=input_config)
    print('Net set up successfully!')
    pytorch_total_params = sum(p.numel() for p in net.parameters())
    pytorch_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("\tNumber of parameters: {}\n\tNumber of trainable parameters: {}".format(pytorch_total_params,
                                                                                    pytorch_trainable_params))
    # TODO: add posibility to load old weights

    #########################
    # Set which device run on
    #########################
    if train_config.use_cuda:
        # "If you load your samples in the Dataset on CPU and would like to push it during training to the GPU,
        # you can speed up the host to device transfer by enabling pin_memory."
        # - ptrblck [https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723]
        pin_memory = True
        if train_config.multi_gpu:
            device_ids = list(range(torch.cuda.device_count()))
            device = torch.device("cuda:" + str(device_ids[0]))
            net = net.to(device)
            net = nn.DataParallel(net)
            print('\nUsing multiple GPUs.\n\tDevices: {}\n\tOutput device: {}\n'.format(device_ids, device))
        else:
            device = torch.device("cuda:" + str(train_config.cuda_device))
            print('\nUsing device {}\n'.format(device))
            net = net.to(device)
    else:
        pin_memory = False
        device = torch.device("cpu")
        print('Using CPU\n')

    #########################
    # Define loss function
    #########################
    loss_func = FafeLoss(input_config, train_config, loss_config, device)
    if train_config.use_cuda:
        loss_func = loss_func.to(device)
        if train_config.multi_gpu:
            loss_func = nn.DataParallel(loss_func)

    #########################
    # Define optimizer
    #########################
    params = list(net.parameters()) + list(loss_func.parameters())
    optimizer = optim.Adam(params,
                           lr=train_config.learning_rate,
                           weight_decay=train_config.weight_decay)
    print('Adams Optimizer set up with\n\tlr = {}\n\twd = {}\n'.format(train_config.learning_rate,
                                                                       train_config.weight_decay))

    #########################
    # Get Datasets
    #########################
    print('Training Data:')
    training_dataloader = DataLoader(ConcatDataset([TemporalBEVsDataset(input_config, root_dir, split='training',
                                                    sequence=seq) for seq in train_config.training_seqs]),
                                     batch_size=train_config.batch_size,
                                     shuffle=train_config.shuffle,
                                     num_workers=train_config.num_workers,
                                     pin_memory=pin_memory)
    print('Validation Data:')
    validation_dataloader = DataLoader(ConcatDataset([TemporalBEVsDataset(input_config, root_dir, split='training',
                                                     sequence=seq) for seq in train_config.validation_seqs]),
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
        train_scaled_L1_mean, train_scaled_euler_mean, train_classification_loss = 0, 0, 0
        eval_scaled_L1_mean, eval_scaled_euler_mean, eval_classification_loss = 0, 0, 0

        #########################
        # TRAINING
        #########################
        tic = time()
        net.train()
        for i_batch, sample_batched in enumerate(training_dataloader):
            input, target, _ = sample_batched

            if train_config.use_cuda:
                input = input.to(device)
                target = target.to(device)

            # Always reset optimizer's gradient each iteration
            optimizer.zero_grad()

            if verbose:
                print('{} i: {} {}'.format('~' * 10, i_batch, '~' * 10))
                print('Input shape: {}'.format(input.shape))
                print('Target shape: {}'.format(target.shape))
                print('Target: {}'.format(target[0][0][0][0]))

            # Forward propagation
            out_detection, out_regression = net.forward(input)
            if verbose:
                print('Output shape det: {}'.format(out_detection.shape))
                print('Output shape reg: {}'.format(out_regression.shape))

            # Calculate the loss
            loss, recall, precision, scaled_l1, scaled_euler, classification_loss = loss_func(out_detection,
                                                                                              out_regression, target,
                                                                                              verbose)
            # Back propagate
            loss.backward()

            # Update the weights
            optimizer.step()

            train_mean_loss += loss
            train_mean_recall += recall
            train_mean_precision += precision
            train_scaled_L1_mean += scaled_l1
            train_scaled_euler_mean += scaled_euler
            train_classification_loss += classification_loss
            train_num_samples += 1

        # Calculate the actual averages
        train_mean_loss /= train_num_samples
        train_mean_recall /= train_num_samples
        train_mean_precision /= train_num_samples
        train_scaled_L1_mean /= train_num_samples
        train_scaled_euler_mean /= train_num_samples
        train_classification_loss /= train_num_samples
        training_time = time() - tic

        #########################
        # EVALUATION
        #########################
        tic2 = time()
        net.eval()
        with torch.no_grad():
            for i_batch, sample_batched in enumerate(validation_dataloader):
                input, target, _ = sample_batched
                if train_config.use_cuda:
                    input = input.to(device)
                    target = target.to(device)
                # Forward propagation
                out_detection, out_regression = net.forward(input)
                # Calculate the loss
                loss, recall, precision, scaled_l1, scaled_euler, classification_loss = loss_func(out_detection,
                                                                                                  out_regression,
                                                                                                  target, verbose)
                eval_mean_loss += loss
                eval_mean_recall += recall
                eval_mean_precision += precision
                eval_scaled_L1_mean += scaled_l1
                eval_scaled_euler_mean += scaled_euler
                eval_classification_loss += classification_loss
                eval_num_samples += 1
        eval_mean_loss /= eval_num_samples
        eval_mean_recall /= eval_num_samples
        eval_mean_precision /= eval_num_samples
        eval_scaled_L1_mean /= eval_num_samples
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
                                          train_scaled_L1_mean,
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
                                          eval_scaled_L1_mean,
                                          eval_scaled_euler_mean,
                                          eval_classification_loss,
                                          eval_mean_recall,
                                          eval_mean_precision))
        if train_config.use_visdom:
            # Visualize Loss
            viz.push_data(epoch, vis,
                          loss_window, sub_loss_window, recall_window, precision_window,
                          train_mean_loss, eval_mean_loss,
                          train_scaled_L1_mean, train_classification_loss,
                          train_scaled_euler_mean, eval_scaled_euler_mean,
                          eval_scaled_L1_mean, eval_classification_loss,
                          train_mean_recall, eval_mean_recall,
                          train_mean_precision, eval_mean_precision)

        #########################
        # SAVE WEIGHTS (every save_weights_modulus th epoch)
        #########################
        if epoch % train_config.save_weights_modulus == 0 or epoch == train_config.max_epochs - 1:
            save_filename = weights_filename + '_epoch_' + str(epoch)
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss},
                save_filename)

    print('Training Complete [' + strftime("%Y-%m-%d %H:%M") + ']')


if __name__ == "__main__":
    train()
