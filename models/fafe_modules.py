from torch import nn


class ResBlock2(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, norm_layer=None):
        super(ResBlock2, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv5x5(in_channels, out_channels, stride)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv5x5(out_channels, out_channels)
        self.bn2 = norm_layer(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out


class ResBlock3(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, norm_layer=None):
        super(ResBlock3, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(in_channels, out_channels)
        self.bn1 = norm_layer(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels, stride)
        self.bn2 = norm_layer(out_channels)
        self.conv3 = conv1x1(out_channels, out_channels)
        self.bn3 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class DetectionHead(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DetectionHead, self).__init__()
        self.conv3_1 = conv3x3(in_channels=in_channels, out_channels=384, stride=1, bias=True)
        self.conv3_2 = conv3x3(in_channels=384, out_channels=out_channels, stride=1, bias=True)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv3_1(x)
        out = self.conv3_2(x)
        # out = self.softmax(out)

        return out


class RegressionHead(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(RegressionHead, self).__init__()
        self.conv3_1 = conv3x3(in_channels=in_channels, out_channels=384, stride=1, bias=True)
        self.conv3_2 = conv3x3(in_channels=384, out_channels=out_channels, stride=1, bias=True)

    def forward(self, x):
        x = self.conv3_1(x)
        out = self.conv3_2(x)

        return out


def conv1x1(in_channels, out_channels, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_channels, out_channels, stride=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=1, bias=bias)


def conv5x5(in_channels, out_channels, stride=1):
    """5x5 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=stride,
                     padding=2, bias=False)
