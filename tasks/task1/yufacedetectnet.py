import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.ops import nms

def combine_conv_bn(conv, bn):
    conv_result = nn.Conv2d(conv.in_channels, conv.out_channels, 
                            kernel_size=conv.kernel_size, stride=conv.stride, 
                            padding=conv.padding, groups = conv.groups, bias=True)
    
    scales = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    conv_result.bias[:] = (conv.bias - bn.running_mean) * scales + bn.bias
    for ch in range(conv.out_channels):
        conv_result.weight[ch, :, :, :] = conv.weight[ch, :, :, :] * scales[ch]

    return conv_result

def convert_param2string(conv, name, is_depthwise=False, isfirst3x3x3=False, precision='.3g'):
    '''
    Convert the weights to strings
    '''
    (out_channels, in_channels, width, height) = conv.weight.size()

    if (isfirst3x3x3):
        w = conv.weight.detach().numpy().reshape((-1,27))
        w_zeros = np.zeros((out_channels ,5))
        w = np.hstack((w, w_zeros))
        w = w.reshape(-1)
    elif (is_depthwise):
        w = conv.weight.detach().numpy().reshape((-1,9)).transpose().reshape(-1)
    else:
        w = conv.weight.detach().numpy().reshape(-1)

    b = conv.bias.detach().numpy().reshape(-1)

    if (isfirst3x3x3):
        lengthstr_w = str(out_channels) + '* 32 * 1 * 1'
        # print(conv.in_channels, conv.out_channels, conv.kernel_size)
    else:
        lengthstr_w = str(out_channels) + '*' + str(in_channels) + '*' + str(width) + '*' + str(height)
    resultstr = 'float ' + name + '_weight[' + lengthstr_w + '] = {'

    for idx in range(w.size - 1):
        resultstr += (format(w[idx], precision) + ',')
    resultstr += (format(w[-1], precision))
    resultstr += '};\n'

    resultstr += 'float ' + name + '_bias[' + str(out_channels) + '] = {'
    for idx in range(b.size - 1):
        resultstr += (format(b[idx], precision) + ',')
    resultstr += (format(b[-1], precision))
    resultstr += '};\n'

    return resultstr
    

class ConvDPUnit(nn.Module):
    def __init__(self, in_channels, out_channels, withBNRelu=True):
        super(ConvDPUnit, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=True, groups=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True, groups=out_channels)
        self.withBNRelu = withBNRelu
        if withBNRelu:
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        if self.withBNRelu:
            x = self.bn(x)
            x = F.relu(x, inplace=True)
        return x

    def convert_to_cppstring(self, varname):
        rs1 = convert_param2string(self.conv1, varname+'_1', False)
        if self.withBNRelu:
            rs2 = convert_param2string(combine_conv_bn(self.conv2, self.bn), varname+'_2', True)
        else:
            rs2 = convert_param2string(self.conv2, varname+'_2', True)

        return rs1 + rs2

class Conv_head(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(Conv_head, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 2, 1, bias=True, groups=1)
        self.conv2 = ConvDPUnit(mid_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(mid_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)

        return x

    def convert_to_cppstring(self, varname):
       rs1 = convert_param2string(self.conv1, varname+'_0', False, True)
       rs2 = self.conv2.convert_to_cppstring(varname)
       return rs1 + rs2

class Conv4layerBlock(nn.Module):
    def __init__(self, in_channels, out_channels, withBNRelu=True):
        super(Conv4layerBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = ConvDPUnit(in_channels, in_channels, True)
        self.conv2 = ConvDPUnit(in_channels, out_channels, withBNRelu)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        return x

class YuFaceDetectNet(nn.Module):

    def __init__(self, phase, size):
        super(YuFaceDetectNet, self).__init__()
        self.phase = phase
        self.num_classes = 2
        self.size = size
        
        self.model0 = Conv_head(3, 16, 16)
        self.model1 = Conv4layerBlock(16, 64)
        self.model2 = Conv4layerBlock(64, 64)
        self.model3 = Conv4layerBlock(64, 64)
        self.model4 = Conv4layerBlock(64, 64)
        self.model5 = Conv4layerBlock(64, 64)
        self.model6 = Conv4layerBlock(64, 64)

        self.head = self.multibox(self.num_classes)

        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)

        if self.phase == 'train':
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

    def multibox(self, num_classes):
        head_layers = []
        head_layers += [Conv4layerBlock(self.model3.out_channels, 3 * (14 + 2 + 1), False)]
        head_layers += [Conv4layerBlock(self.model4.out_channels, 2 * (14 + 2 + 1), False)]
        head_layers += [Conv4layerBlock(self.model5.out_channels, 2 * (14 + 2 + 1), False)]
        head_layers += [Conv4layerBlock(self.model6.out_channels, 3 * (14 + 2 + 1), False)]
        return nn.Sequential(*head_layers)

    def forward(self, x):
        #num, ch, height, width = x.size()

        detection_sources = list()
        head_data = list()
        loc_data = list()
        conf_data = list()
        iou_data = list()

        x = self.model0(x)
        x = F.max_pool2d(x, 2)
        x = self.model1(x)
        x = self.model2(x)
        x = F.max_pool2d(x, 2)
        x = self.model3(x)
        detection_sources.append(x)

        x = F.max_pool2d(x, 2)
        x = self.model4(x)
        detection_sources.append(x)

        x = F.max_pool2d(x, 2)
        x = self.model5(x)
        detection_sources.append(x)

        x = F.max_pool2d(x, 2)
        x = self.model6(x)
        detection_sources.append(x)

        for (x, h) in zip(detection_sources, self.head):
            head_data.append(h(x).permute(0, 2, 3, 1).contiguous())

        head_data = torch.cat([o.view(o.size(0), -1) for o in head_data], 1)
        head_data = head_data.view(head_data.size(0), -1, 17)

        loc_data = head_data[:, :, 0:14]
        conf_data = head_data[:, :, 14:16]
        iou_data = head_data[:,:, 16:17]
        output = (loc_data, conf_data, iou_data)
        # print ('output size', head_data[0].size())

        if self.phase == "test":
            loc_data = loc_data.view(-1, 14)
            conf_data = self.softmax(conf_data.view(-1, self.num_classes))
            iou_data = iou_data.view(-1, 1)
            output = (loc_data, conf_data, iou_data)
        else:
            output = (loc_data.view(loc_data.size(0), -1, 14),
                    conf_data.view(conf_data.size(0), -1, self.num_classes),
                    iou_data.view(iou_data.size(0), -1, 1))

        return output


    def export_cpp(self, filename):
        '''This function can export CPP data file for libfacedetection'''
        result_str = '// Auto generated data file\n'
        result_str += '// Copyright (c) 2018-2021, Shiqi Yu, all rights reserved.\n'
        result_str += '#include "facedetectcnn.h" \n\n'

        DPConvs = []
        DPConvs += [self.model0.conv1, self.model0.conv2]
        DPConvs += [self.model1.conv1, self.model1.conv2]
        DPConvs += [self.model2.conv1, self.model2.conv2]
        DPConvs += [self.model3.conv1, self.model3.conv2]
        DPConvs += [self.model4.conv1, self.model4.conv2]
        DPConvs += [self.model5.conv1, self.model5.conv2]
        DPConvs += [self.model6.conv1, self.model6.conv2]
        # for (l, c) in zip(self.loc, self.conf):
        #     DPConvs += [l.conv1, l.conv2]
        #     DPConvs += [c.conv1, c.conv2]
        for layers in self.head:
            DPConvs += [layers.conv1, layers.conv2]

        # convert to a string
        num_conv = len(DPConvs)

        # the first 3x3x3 conv layer
        result_str += convert_param2string(combine_conv_bn(DPConvs[0], self.model0.bn1), 'f0', False, True)

        # the rest depthwise+pointwise conv layers
        for idx in range(1, num_conv):
            rs = DPConvs[idx].convert_to_cppstring('f' + str(idx))
            result_str += rs
            result_str += '\n'

        result_str += 'ConvInfoStruct param_pConvInfo[' + str(num_conv*2 - 1) + '] = { \n'
        result_str += '   {32, ' + str(DPConvs[0].out_channels) +', false, true, true, f0_weight, f0_bias},\n'
        
        for idx in range(1, num_conv):
            result_str += ('    {' +
                           str(DPConvs[idx].in_channels) + ', ' +
                           str(DPConvs[idx].out_channels) + ', ' +
                           'false, ' + # is_depthwise 
                           'true, ' + # is_pointwise
                           'false, ' + # with_relu
                           'f' + str(idx) + '_1_weight' + ', ' +
                           'f' + str(idx) + '_1_bias' +
                           '}')

            result_str += ','
            result_str += '\n'

            with_relu = 'false, '
            if(DPConvs[idx].withBNRelu):
                with_relu = 'true, '

            result_str += ('    {' +
                           str(DPConvs[idx].out_channels) + ', ' +
                           str(DPConvs[idx].out_channels) + ', ' +
                           'true, ' + # is_depthwise 
                           'false, ' + # is_pointwise
                           with_relu + # with_relu
                           'f' + str(idx) + '_2_weight' + ', ' +
                           'f' + str(idx) + '_2_bias' +
                           '}')

            if (idx < num_conv - 1):
                result_str += ','
            result_str += '\n'

        result_str += '};\n'
        

        # write the content to a file
        #print(result_str)
        with open(filename, 'w') as f:
            f.write(result_str)
            f.close()

        return 0 
