from torch import nn
from .fafe_modules import ResBlock2, ResBlock3, RegressionHead, DetectionHead, conv1x1


class LittleFafe(nn.Module):
    def __init__(self, input_config):
        super().__init__()

        # Check if PP or BEV input
        if input_config.use_pp:
            in_channels = input_config.pp_num_filters[-1] * input_config.num_conseq_frames
        else:
            in_channels = input_config.bev_shape[2] * input_config.num_conseq_frames

        # BACKBONE NETWORK
        self.layer1 = self._make_layer(ResBlock2, in_channels=in_channels, out_channels=160, stride=2)
        self.layer2 = self._make_layer(ResBlock2, in_channels=160, out_channels=192, stride=1)
        self.layer3 = self._make_layer(ResBlock3, in_channels=192, out_channels=224, stride=2)

        # DETECTION HEAD
        # output: [p_0, p_i, ..., p_num_classes] \forall i = object_class
        num_det = (input_config.num_classes + 1) * input_config.num_anchors * input_config.num_conseq_frames
        self.detection_head = DetectionHead(in_channels=224, out_channels=num_det)

        # REGRESSION HEAD
        # output: [t_x, t_y, t_w, t_l, t_re, t_im] \forall anchors  \forall conseq_frames
        num_regr = input_config.num_reg_targets * input_config.num_anchors * input_config.num_conseq_frames
        self.regression_head = RegressionHead(in_channels=224, out_channels=num_regr)

    def _make_layer(self, block, in_channels, out_channels, stride, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(conv1x1(in_channels, out_channels, stride),
                                       norm_layer(out_channels))
        return block(in_channels, out_channels, stride, downsample, norm_layer)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = self.layer4(out)
        # out = out.reshape(out.size(0), -1)

        out_detection = self.detection_head(out)
        out_regression = self.regression_head(out)

        return out_detection, out_regression
