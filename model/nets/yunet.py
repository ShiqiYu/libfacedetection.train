import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import Conv_head, Conv4layerBlock


class Yunet(nn.Module):
    def __init__(self, cfg_layers, activation_type='relu'):
        super().__init__()

        self.model0 = Conv_head(*cfg_layers[0], activation_type=activation_type)
        for i in range(1, len(cfg_layers)):
            self.add_module(f'model{i}', Conv4layerBlock(*cfg_layers[i], activation_type=activation_type))
        # self.model1 = Conv4layerBlock(16, 64, activation_type=activation_type)
        # self.model2 = Conv4layerBlock(64, 64, activation_type=activation_type)
        # self.model3 = Conv4layerBlock(64, 64, activation_type=activation_type)
        # self.model4 = Conv4layerBlock(64, 64, activation_type=activation_type)
        # self.model5 = Conv4layerBlock(64, 64, activation_type=activation_type)
        # self.model6 = Conv4layerBlock(64, 64, activation_type=activation_type)
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

    def forward(self, x):
        x = self.model0(x)
        x = F.max_pool2d(x, 2)
        x = self.model1(x)
        x = self.model2(x)
        x = F.max_pool2d(x, 2)
        p1 = self.model3(x)
        x = F.max_pool2d(p1, 2)
        p2 = self.model4(x)
        x = F.max_pool2d(p2, 2)
        p3 = self.model5(x)
        x = F.max_pool2d(p3, 2)
        p4 = self.model6(x)
        
        return [p1, p2, p3, p4]