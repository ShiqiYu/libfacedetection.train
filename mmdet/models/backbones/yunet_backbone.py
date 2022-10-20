import torch.nn as nn
import torch.nn.functional as F

from ..builder import BACKBONES
from ..utils.yunet_layer import Conv4layerBlock, Conv_head


@BACKBONES.register_module()
class YuNetBackbone(nn.Module):

    def __init__(self, stage_channels, downsample_idx, out_idx):
        super().__init__()
        self.layer_num = len(stage_channels)
        self.downsample_idx = downsample_idx
        self.out_idx = out_idx
        self.model0 = Conv_head(*stage_channels[0])
        for i in range(1, self.layer_num):
            self.add_module(f'model{i}', Conv4layerBlock(*stage_channels[i]))
        self.init_weights()

    def init_weights(self, pretrained=None):
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

    def forward(self, x):
        out = []
        for i in range(self.layer_num):
            x = self.__getattr__(f'model{i}')(x)
            if i in self.out_idx:
                out.append(x)
            if i in self.downsample_idx:
                x = F.max_pool2d(x, 2)
        return out
