from torch import nn
from models.fafenet import FafeNet
from loss.loss import FafeLoss
from models.fafepillar import PillarOfFafe


class HaveFafe(nn.Module):
    def __init__(self, input_config, train_config, loss_config):
        super().__init__()

        self.net = FafeNet(input_config=input_config)
        self.loss = FafeLoss(input_config, train_config, loss_config)

    def __call__(self, example, target, verbose):
        return self.forward(example, target, verbose)

    def forward(self, example, target, verbose):
        #########################
        # Forward propagation
        #########################
        out_detection, out_regression = self.net.forward(example)

        #########################
        # Calculate the loss
        #########################
        loss, recall, precision, scaled_l1, scaled_euler, classification_loss = self.loss(out_detection,
                                                                                          out_regression,
                                                                                          target,
                                                                                          verbose)
        return loss, recall, precision, scaled_l1, scaled_euler, classification_loss


class HaveFafePillar(nn.Module):
    def __init__(self, input_config, train_config, loss_config):
        super().__init__()

        self.pp = PillarOfFafe(input_config=input_config,
                               batch_size=train_config.batch_size,
                               verbose=train_config.verbose)

        self.net = FafeNet(input_config=input_config)

        self.loss = FafeLoss(input_config, train_config, loss_config)

    def __call__(self, example, target, verbose):
        return self.forward(example, target, verbose)

    def forward(self, example, target, verbose):
        #########################
        # Forward propagation
        #########################
        psuedo_img = self.pillar(example)
        out_detection, out_regression = self.net.forward(psuedo_img)

        #########################
        # Calculate the loss
        #########################
        loss, recall, precision, scaled_l1, scaled_euler, classification_loss = self.loss(out_detection,
                                                                                          out_regression,
                                                                                          target,
                                                                                          verbose)
        return loss, recall, precision, scaled_l1, scaled_euler, classification_loss


class HaveFafe4Eval(nn.Module):
    def __init__(self, input_config):
        super().__init__()
        self.net = FafeNet(input_config=input_config)

    def __call__(self, example, *args, **kwargs):
        return self.forward(example)

    def forward(self, example):
        out_detection, out_regression = self.net.forward(example)
        return out_detection, out_regression
