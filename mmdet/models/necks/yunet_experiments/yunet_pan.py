import torch.nn as nn
import torch.nn.functional as F

from ...builder import NECKS
# from mmcv.runner import auto_fp16
from ...utils.yunet_layer import ConvDPUnit


@NECKS.register_module()
class WWHead_PAN(nn.Module):

    def __init__(self, in_channels, lateral_channel, out_idx):
        super().__init__()
        self.num_layers = len(in_channels)
        self.out_idx = out_idx
        self.lateral_convs = nn.ModuleList()
        self.merge_convs = nn.ModuleList()
        self.pan_convs = nn.ModuleList()
        self.downsample = nn.ModuleList()
        self.head_convs = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        for i in range(self.num_layers):
            if i != 0 or i != self.num_layers - 2:
                self.merge_convs.append(
                    ConvDPUnit(lateral_channel, lateral_channel))
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels[i], lateral_channel, 1, 1, 0),
                    nn.BatchNorm2d(32), nn.ReLU()))
            self.pan_convs.append(ConvDPUnit(lateral_channel, lateral_channel))
            if i != self.num_layers - 1:
                self.downsample.append(nn.MaxPool2d(2))
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.xavier_normal_(m.weight.data)
                    m.bias.data.fill_(0.02)
                else:
                    m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # @auto_fp16()
    def forward(self, feats):
        num_feats = len(feats)

        # lateral flow
        lateral_feats = [
            lateral_conv(x)
            for lateral_conv, x in zip(self.lateral_convs, feats)
        ]

        # top-down flow
        for i in range(num_feats - 1, 0, -1):
            lateral_feats[i - 1] = lateral_feats[i - 1] + F.interpolate(
                lateral_feats[i],
                size=lateral_feats[i - 1].shape[-2:],
                mode='nearest')

        # lateral flow
        for i in range(1, num_feats - 1):
            lateral_feats[i] = self.merge_convs[i](lateral_feats[i])

        # bottom-up flow
        for i in range(num_feats):
            feat = self.pan_convs[i](lateral_feats[i])
            if i != num_feats - 1:
                lateral_feats[i +
                              1] = lateral_feats[i + 1] + self.downsample[i](
                                  feat)
        outs = [lateral_feats[i] for i in self.out_idx]
        return outs
