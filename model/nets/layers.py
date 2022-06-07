import torch.nn as nn
import torch
import numpy as np

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


def get_activation_fn(activation_type, inplace=True):
    
    if activation_type == 'relu':
        relu = nn.ReLU(inplace=inplace)
    elif activation_type == "swish":
        relu = nn.SiLU(inplace=inplace)
    elif activation_type == "hardswish":
        relu = nn.Hardswish(inplace=inplace)
    else:
        raise ValueError(f'Unknown activation type: {activation_type}')
    return relu

class ConvDPUnit(nn.Module):
    def __init__(self, in_channels, out_channels, withBNRelu=True, activation_type='relu'):
        super(ConvDPUnit, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=True, groups=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True, groups=out_channels)
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

    def convert_to_cppstring(self, varname):
        rs1 = convert_param2string(self.conv1, varname+'_1', False)
        if self.withBNRelu:
            rs2 = convert_param2string(combine_conv_bn(self.conv2, self.bn), varname+'_2', True)
        else:
            rs2 = convert_param2string(self.conv2, varname+'_2', True)

        return rs1 + rs2

class Conv_head(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, activation_type='relu'):
        super(Conv_head, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 2, 1, bias=True, groups=1)
        self.conv2 = ConvDPUnit(mid_channels, out_channels, True, activation_type=activation_type)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = get_activation_fn(activation_type)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        return x

    def convert_to_cppstring(self, varname):
       rs1 = convert_param2string(combine_conv_bn(self.conv1, self.bn1), varname + '0', False, True)
       rs2 = self.conv2.convert_to_cppstring(varname + '1')
       return rs1 + rs2 + '\n'

class Conv4layerBlock(nn.Module):
    def __init__(self, in_channels, out_channels, withBNRelu=True, activation_type='relu'):
        super(Conv4layerBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = ConvDPUnit(in_channels, in_channels, True, activation_type)
        self.conv2 = ConvDPUnit(in_channels, out_channels, withBNRelu, activation_type)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x



from itertools import product as product
class PriorBox(object):
    def __init__(self, min_sizes, steps, clip, ratio):
        super(PriorBox, self).__init__()
        self.min_sizes = min_sizes
        self.steps = steps
        self.clip = clip
        self.ratio = ratio
    def __call__(self, image_size):
        feature_map_2th = [int(int((image_size[0] + 1) / 2) / 2),
                                int(int((image_size[1] + 1) / 2) / 2)]
        feature_map_3th = [int(feature_map_2th[0] / 2),
                                int(feature_map_2th[1] / 2)]
        feature_map_4th = [int(feature_map_3th[0] / 2),
                                int(feature_map_3th[1] / 2)]
        feature_map_5th = [int(feature_map_4th[0] / 2),
                                int(feature_map_4th[1] / 2)]
        feature_map_6th = [int(feature_map_5th[0] / 2),
                                int(feature_map_5th[1] / 2)]

        feature_maps = [feature_map_3th, feature_map_4th,
                             feature_map_5th, feature_map_6th]
        anchors = []
        for k, f in enumerate(feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    cx = (j + 0.5) * self.steps[k] / image_size[1]
                    cy = (i + 0.5) * self.steps[k] / image_size[0]
                    for r in self.ratio:
                        s_ky = min_size / image_size[0]
                        s_kx = r * min_size / image_size[1]
                        anchors += [cx, cy, s_kx, s_ky]
        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output