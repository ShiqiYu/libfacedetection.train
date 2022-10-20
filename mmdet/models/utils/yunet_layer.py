import torch.nn as nn


def get_activation_fn(activation_type, inplace=True):
    if activation_type == 'relu':
        relu = nn.ReLU(inplace=inplace)
    elif activation_type == 'swish':
        relu = nn.SiLU(inplace=inplace)
    elif activation_type == 'hardswish':
        relu = nn.Hardswish(inplace=inplace)
    else:
        raise ValueError(f'Unknown activation type: {activation_type}')
    return relu


class ConvDPUnit(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 withBNRelu=True,
                 activation_type='relu'):
        super(ConvDPUnit, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 1, 1, 0, bias=True, groups=1)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            1,
            1,
            bias=True,
            groups=out_channels)
        self.withBNRelu = withBNRelu
        if withBNRelu:
            self.bn = nn.BatchNorm2d(out_channels)
            self.relu = get_activation_fn(activation_type)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        if self.withBNRelu:
            x = self.bn(x)
            x = self.relu(x)
        return x


class Conv_head(nn.Module):

    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 activation_type='relu'):
        super(Conv_head, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(
            in_channels, mid_channels, 3, 2, 1, bias=True, groups=1)
        self.conv2 = ConvDPUnit(
            mid_channels, out_channels, True, activation_type=activation_type)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = get_activation_fn(activation_type)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        return x


class Conv4layerBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 withBNRelu=True,
                 activation_type='relu'):
        super(Conv4layerBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = ConvDPUnit(in_channels, in_channels, True,
                                activation_type)
        self.conv2 = ConvDPUnit(in_channels, out_channels, withBNRelu,
                                activation_type)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
