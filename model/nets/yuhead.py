from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

# sys.path.append('/home/ww/projects/yudet/model/nets/')
from .layers import Conv4layerBlock, ConvDPUnit, get_activation_fn


def build_head(name, in_channels, out_channels, activation_type='relu'):
    if name is None:
        head_conv = Yuhead
    head_conv = globals().get(name, None)
    if head_conv is None:
        raise ImportError(f"No head name {name}")
    return head_conv(in_channels, out_channels, activation_type)

# just apply silu to yuhead
class Yuhead(nn.Module):
    def __init__(self, in_channels, out_channels, activation_type='relu'):
        super().__init__()
        assert len(in_channels) == len(out_channels)
        self.head = nn.ModuleList(
            [Conv4layerBlock(in_c, out_c, withBNRelu=False, activation_type=activation_type) for \
                in_c, out_c in zip(in_channels, out_channels)]
        )
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
        assert isinstance(feats, List)
        outs = []
        up = self.head[-1].conv1(feats[-1])
        out = self.head[-1].conv2(up)
        outs.append(out)
        for i in range(len(feats) - 2, -1, -1):
            up = F.interpolate(
                up, 
                size=[feats[i].size(2), feats[i].size(3)], 
                mode="nearest"
            )
            up = self.head[i].conv1(feats[i] + up)
            out = self.head[i].conv2(up)
            outs.insert(0, out)

        return outs
        
class Yuhead_PAN(nn.Module):
    def __init__(self, in_channels, out_channels, activation_type):
        super().__init__()
        num_layers = len(in_channels)
        self.lateral_convs = nn.ModuleList()
        self.merge_convs = nn.ModuleList()
        self.pan_convs = nn.ModuleList()
        self.head = nn.ModuleList()
        self.downsample = nn.ModuleList()
        for i, (in_c, out_c) in enumerate(zip(in_channels, out_channels)):
            if i != 0 or i != num_layers - 2:
                self.merge_convs.append(ConvDPUnit(32, 32, activation_type))
            self.lateral_convs.append(
                nn.Sequential(nn.Conv2d(in_c, 32, 1, 1, 0), 
                nn.BatchNorm2d(32), 
                get_activation_fn(activation_type)))
            self.pan_convs.append(ConvDPUnit(32, 32, activation_type))
            self.head.append(ConvDPUnit(32, out_c, False))
            if i != num_layers - 1:
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

    def forward(self, feats):
        assert isinstance(feats, List)
        num_feats = len(feats)
        
        # lateral flow
        lateral_feats = [lateral_conv(x) for lateral_conv, x in zip(self.lateral_convs, feats)]

        # top-down flow
        for i in range(num_feats - 1, 0, -1):
            lateral_feats[i - 1] = lateral_feats[i - 1] + F.interpolate(lateral_feats[i], size=lateral_feats[i - 1].shape[-2:], mode='nearest')

        # lateral flow
        for i in range(1, num_feats - 1):
            lateral_feats[i] = self.merge_convs[i](lateral_feats[i])
        
        outs = []
        # bottom-up flow
        for i in range(num_feats):
            feat = self.pan_convs[i](lateral_feats[i])
            if i != num_feats - 1:
                lateral_feats[i + 1] = lateral_feats[i + 1] + self.downsample[i](feat)
            outs.append(self.head[i](feat))

        return outs

class Yuhead_double(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        assert len(in_channels) == len(out_channels)

        self.mix = nn.ModuleList(
            [ConvDPUnit(in_c, in_c, withBNRelu=True) for \
                in_c in in_channels]
        )
        self.head_cls = nn.ModuleList(
            [ConvDPUnit(in_c, int(out_c / 17) * 2, withBNRelu=False) for \
                in_c, out_c in zip(in_channels, out_channels)]
        )
        self.head_reg = nn.ModuleList(
            [ConvDPUnit(in_c, int(out_c / 17) * 15, withBNRelu=False) for \
                in_c, out_c in zip(in_channels, out_channels)]
        )
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
        assert isinstance(feats, List)
        outs_cls = []
        outs_reg = []
        up = self.mix[-1](feats[-1])
        outs_cls.append(self.head_cls[-1](up))
        outs_reg.append(self.head_reg[-1](up))
        for i in range(len(feats) - 2, -1, -1):
            up = F.interpolate(
                up, 
                size=[feats[i].size(2), feats[i].size(3)], 
                mode="nearest"
            )
            up = self.mix[i](feats[i] + up)
            outs_cls.insert(0, self.head_cls[i](up))
            outs_reg.insert(0, self.head_reg[i](up))

        outs = []
        for cls, reg in zip(outs_cls, outs_reg):
            n, c, h, w = cls.shape

            cls_data = cls.permute(0, 2, 3, 1).view(n, h, w, -1, 2)
            reg_data = reg.permute(0, 2, 3, 1).view(n, h, w, -1, 15)
            out = torch.cat([reg_data[..., :-1], cls_data[..., :], reg_data[..., -1:]], dim=-1)
            outs.append(out.view(n, h, w, -1).permute(0, 3, 1, 2).contiguous())
        return outs


class Yuhead_originfpn(nn.Module):
    def __init__(self, in_channels, out_channels, activation_type='relu'):
        super().__init__()

        assert len(in_channels) == len(out_channels)
        self.head = nn.ModuleList(
            [Conv4layerBlock(in_c, out_c, withBNRelu=False, activation_type=activation_type) for \
                in_c, out_c in zip(in_channels, out_channels)]
        )
        self.fpn = nn.ModuleList([nn.Sequential(
                                nn.Conv2d(
                                            in_channels=in_c,
                                            out_channels=in_c,
                                            stride=1,
                                            kernel_size=1),
                                nn.BatchNorm2d(in_c),
                                get_activation_fn(activation_type)           
                                ) for in_c in in_channels])
                
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
        assert isinstance(feats, List)
        outs = []
        up = self.head[-1].conv1(self.fpn[-1](feats[-1]))
        out = self.head[-1].conv2(up)
        outs.append(out)
        for i in range(len(feats) - 2, -1, -1):
            up = F.interpolate(
                up, 
                size=[feats[i].size(2), feats[i].size(3)], 
                mode="nearest"
            )
            up = self.head[i].conv1(self.fpn[i](feats[i]) + up)
            out = self.head[i].conv2(up)
            outs.insert(0, out)

        return outs   

class Yuhead_originfpn_large(nn.Module):
    def __init__(self, in_channels, out_channels, activation_type='relu'):
        super().__init__()

        assert len(in_channels) == len(out_channels)
        self.head = nn.ModuleList(
            [ConvDPUnit(in_c, out_c, withBNRelu=False) for \
                in_c, out_c in zip(in_channels, out_channels)]
        )
        self.fpn_pre = nn.ModuleList([nn.Sequential(
                                nn.Conv2d(
                                            in_channels=in_c,
                                            out_channels=in_c,
                                            stride=1,
                                            kernel_size=1),
                                nn.BatchNorm2d(in_c),
                                get_activation_fn(activation_type)          
                                ) for in_c in in_channels])
        self.fpn_aft = nn.ModuleList([nn.Sequential(
                                nn.Conv2d(
                                            in_channels=in_c,
                                            out_channels=in_c,
                                            stride=1,
                                            kernel_size=3,
                                            padding=1),
                                nn.BatchNorm2d(in_c),
                                get_activation_fn(activation_type)           
                                ) for in_c in in_channels])             
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
        assert isinstance(feats, List)
        outs = []
        up = self.fpn_aft[-1](self.fpn_pre[-1](feats[-1]))
        out = self.head[-1](up)
        outs.append(out)
        for i in range(len(feats) - 2, -1, -1):
            up = F.interpolate(
                up, 
                size=[feats[i].size(2), feats[i].size(3)], 
                mode="nearest"
            )
            up = self.fpn_aft[i](self.fpn_pre[i](feats[i]) + up)
            out = self.head[i](up)
            outs.insert(0, out)

        return outs 


class Yuhead_naive(nn.Module):
    def __init__(self, in_channels, out_channels, activation_type='relu'):
        super().__init__()
        assert len(in_channels) == len(out_channels)
        self.head = nn.ModuleList(
            [Conv4layerBlock(in_c, out_c, withBNRelu=False, activation_type=activation_type) for \
                in_c, out_c in zip(in_channels, out_channels)]
        )
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
        assert isinstance(feats, List)
        outs = [h(f) for h, f in zip(self.head, feats)]
        return outs


class WWHead_PAN(nn.Module):
    def __init__(self, in_channels, out_channels, activation_type='relu'):
        super().__init__()
        num_layers = len(in_channels)
        self.lateral_convs = nn.ModuleList()
        self.merge_convs = nn.ModuleList()
        self.pan_convs = nn.ModuleList()
        self.downsample = nn.ModuleList()
        self.head_convs = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        for i, (in_c, out_c) in enumerate(zip(in_channels, out_channels)):
            if i != 0 or i != num_layers - 2:
                self.merge_convs.append(ConvDPUnit(32, 32, activation_type))
            self.lateral_convs.append(
                nn.Sequential(nn.Conv2d(in_c, 32, 1, 1, 0), 
                nn.BatchNorm2d(32), 
                get_activation_fn(activation_type)))
            self.pan_convs.append(ConvDPUnit(32, 32, activation_type))
            self.head_convs.append(ConvDPUnit(32, 64))
            self.cls_convs.append(ConvDPUnit(64, out_c, False))
            if i != num_layers - 1:
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
        bias_cls = -4.595
        for m in self.cls_convs:   
            for sub_m in m.modules():          
                if isinstance(sub_m, nn.Conv2d):
                    if sub_m.bias is not None:
                        sub_m.bias.data.fill_(bias_cls)




            
    def forward(self, feats):
        assert isinstance(feats, List)
        num_feats = len(feats)
        
        # lateral flow
        lateral_feats = [lateral_conv(x) for lateral_conv, x in zip(self.lateral_convs, feats)]

        # top-down flow
        for i in range(num_feats - 1, 0, -1):
            lateral_feats[i - 1] = lateral_feats[i - 1] + F.interpolate(lateral_feats[i], size=lateral_feats[i - 1].shape[-2:], mode='nearest')

        # lateral flow
        for i in range(1, num_feats - 1):
            lateral_feats[i] = self.merge_convs[i](lateral_feats[i])
        
        outs = []
        # bottom-up flow
        for i in range(num_feats):
            feat = self.pan_convs[i](lateral_feats[i])
            if i != num_feats - 1:
                lateral_feats[i + 1] = lateral_feats[i + 1] + self.downsample[i](feat)
            cls_feat = self.head_convs[i](feat)
            outs.append(self.cls_convs[i](cls_feat))

        return outs




if __name__ == "__main__":
    net = Yuhead_PAN(
        in_channels=[64, 64, 64, 64],
        out_channels=[51, 34, 34, 51],
        activation_type='relu'
    )
    feats = [
        torch.rand(1, 64, 80, 80),
        torch.rand(1, 64, 40, 40),
        torch.rand(1, 64, 20, 20),
        torch.rand(1, 64, 10, 10),
    ]
    x = net(feats)


