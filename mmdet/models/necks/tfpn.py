import torch.nn as nn
import torch.nn.functional as F

from ..builder import NECKS
from ..utils.yunet_layer import ConvDPUnit


@NECKS.register_module()
class TFPN(nn.Module):

    def __init__(self, in_channels, out_idx):
        super().__init__()
        self.num_layers = len(in_channels)
        self.out_idx = out_idx
        self.lateral_convs = nn.ModuleList()
        for i in range(self.num_layers):
            self.lateral_convs.append(
                ConvDPUnit(in_channels[i], in_channels[i], True))
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

    def forward(self, feats):
        num_feats = len(feats)

        # top-down flow
        for i in range(num_feats - 1, 0, -1):
            feats[i] = self.lateral_convs[i](feats[i])
            feats[i - 1] = feats[i - 1] + F.interpolate(
                feats[i], scale_factor=2., mode='nearest')

        feats[0] = self.lateral_convs[0](feats[0])

        outs = [feats[i] for i in self.out_idx]
        return outs
