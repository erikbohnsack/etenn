import unittest
from models.fafenet import FafeNet
from loss.loss import FafeLoss
import torch
import torch.optim as optim
from train import train
from cfg.config import TrainConfig, InputConfig
import os
from fafe_utils.kitti_dataset import TemporalBEVsDataset
from torch.utils.data import ConcatDataset, DataLoader
from cfg.config_stuff import load_config, get_root_dir
from time import time, strftime


class TestTrain(unittest.TestCase):
    def test_train(self):
        train()

    def test_train_with_model(self):

        root_dir = get_root_dir()

        if os.path.exists('/Users/erikbohnsack/Code/MOT/fafe/trained_models/weights_2019-04-12_13_43_epoch_85'):
            model_path = '/Users/erikbohnsack/Code/MOT/fafe/trained_models/weights_2019-04-12_13_43_epoch_85'
            config_path = 'cfg/cfg_mac.yml'
        else:
            model_path = '/home/mlt/mot/fafe/trained_models/test_train_2019-04-11_14_01_epoch_40'
            config_path = 'cfg/cfg_mini.yml'

        config = load_config(config_path)

        input_config = InputConfig(config['INPUT_CONFIG'])
        train_config = TrainConfig(config['TRAIN_CONFIG'])

        verbose = train_config.verbose

        net = FafeNet(input_config)
        net.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage)["model_state_dict"])

        #########################
        # Set which device run on
        #########################
        if train_config.use_cuda:
            if train_config.cuda_device == 0:
                device = torch.device("cuda:0")
                print('Using CUDA:{}\n'.format(0))
            elif train_config.cuda_device == 1:
                device = torch.device("cuda:1")
                print('Using CUDA:{}\n'.format(1))
            else:
                print('Functionality for CUDA device cuda:{} not yet implemented.'.format(train_config.cuda_device))
                print('Using cuda:0 instead...\n')
                device = torch.device("cuda:0")
            net = net.to(device)
        else:
            device = torch.device("cpu")
            print('Using CPU\n')

        #########################
        # Define optimizer
        #########################
        optimizer = optim.Adam(net.parameters(),
                               lr=train_config.learning_rate,
                               weight_decay=train_config.weight_decay)
        print('Adams Optimizer set up with\n\tlr = {}\n\twd = {}\n'.format(train_config.learning_rate,
                                                                           train_config.weight_decay))

        loss_func = FafeLoss(input_config, train_config, device)

        training_datasets = [TemporalBEVsDataset(input_config, root_dir, split='training', sequence=seq) for seq in
                             train_config.training_seqs]
        validation_datasets = [TemporalBEVsDataset(input_config, root_dir, split='training', sequence=seq) for seq in
                               train_config.validation_seqs]  # Split still training due to data structure

        training_dataset = ConcatDataset(training_datasets)
        validation_dataset = ConcatDataset(validation_datasets)

        training_dataloader = DataLoader(training_dataset,
                                         batch_size=train_config.batch_size,
                                         shuffle=train_config.shuffle,
                                         num_workers=train_config.num_workers)
        validation_dataloader = DataLoader(validation_dataset,
                                           batch_size=train_config.batch_size,
                                           shuffle=train_config.shuffle,
                                           num_workers=train_config.num_workers)
        ###############################
        # Start training and evaluation
        ###############################
        print('\nTraining initiated [' + strftime("%Y-%m-%d %H:%M") + ']')
        for epoch in range(train_config.max_epochs):
            train_mean_loss, train_mean_recall, train_mean_precision, train_num_samples = 0, 0, 0, 0
            eval_mean_loss, eval_mean_recall, eval_mean_precision, eval_num_samples = 0, 0, 0, 0
            train_scaled_L1_mean, train_classification_loss = 0, 0
            eval_scaled_L1_mean, eval_classification_loss = 0, 0

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
                loss, recall, precision, scaled_L1, scaled_euler, classification_loss = loss_func(out_detection,
                                                                                                  out_regression,
                                                                                                  target, verbose)

                # Back propagate
                loss.backward()

                # Update the weights
                optimizer.step()

                train_mean_loss += loss
                train_mean_recall += recall
                train_mean_precision += precision
                train_scaled_L1_mean += scaled_L1
                train_classification_loss += classification_loss
                train_num_samples += 1
            # Calculate the actual averages
            train_mean_loss /= train_num_samples
            train_mean_recall /= train_num_samples
            train_mean_precision /= train_num_samples
            train_scaled_L1_mean /= train_num_samples
            train_classification_loss /= train_num_samples
            training_time = time() - tic

    def test_concat_dataset(self):
        config_path = 'cfg_mini.yml'
        config = load_config(config_path)

        input_config = InputConfig(config['INPUT_CONFIG'])
        train_config = TrainConfig(config['TRAIN_CONFIG'])

        if os.path.exists("/Users/erikbohnsack/data/"):
            root_dir = "/Users/erikbohnsack/data/"
            sequence = 0
        else:
            root_dir = "/home/mlt/data/"
            sequence = 3

        training_datasets = [TemporalBEVsDataset(input_config, root_dir, split='training', sequence=seq) for seq in
                             train_config.training_seqs]
        validation_datasets = [TemporalBEVsDataset(input_config, root_dir, split='training', sequence=seq) for seq in
                               train_config.validation_seqs]  # Split still training due to data structure

        training_dataset = ConcatDataset(training_datasets)
        validation_dataset = ConcatDataset(validation_datasets)

        training_dataloader = DataLoader(training_dataset,
                                         batch_size=train_config.batch_size,
                                         shuffle=train_config.shuffle,
                                         num_workers=train_config.num_workers)
        validation_dataloader = DataLoader(validation_dataset,
                                           batch_size=train_config.batch_size,
                                           shuffle=train_config.shuffle,
                                           num_workers=train_config.num_workers)
        print("----------------------- TRAINING -----------------------")
        for i_batch, batch_sample in enumerate(training_dataloader):
            print(i_batch)
            input, target, info = batch_sample
            print("Input shape: {}".format(input.shape))
            print("Sequence: {}, real index: {}".format(info["sequence"], info["Current_index"]))
        print("----------------------- VALIDATION -----------------------")
        for i_batch, batch_sample in enumerate(validation_dataloader):
            print(i_batch)
            input, target, info = batch_sample
            print("Input shape: {}".format(input.shape))
            print("Sequence: {}, real index: {}".format(info["sequence"], info["Current_index"]))


if __name__ == "__main__":
    a = TestTrain()
    a.test_train_with_model()
