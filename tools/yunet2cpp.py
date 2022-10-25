import torch.nn as nn
import torch
import numpy as np
from mmdet.core.export import build_model_from_cfg
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert wwfacedet models to libfacedetect dnn data')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--output-file', type=str, default='./work_dirs/facedetectcnn-data.cpp')
    parser.add_argument('--no_summary', action='store_true', help="Output the flops and params")
    args = parser.parse_args()
    return args

def data2str_as_precision(data, precision):
    s = format(data, precision)
    if(s.count('.') == 0 and s.count('e') == 0):
        return s + '.f'
    else:
        return s + 'f'

class CppConvertor(object):
    '''This class can export CPP data file for libfacedetection'''

    def __init__(self, model):
        model.eval()
        self.support_modules = ["Conv_head", "ConvDPUnit", "Conv4layerBlock"]
        self.module_list = []
        self.loop_search_modules(model)
        self.cppdata = []
        self.data = \
"""// Auto generated data file
// Copyright (c) 2018-2023, Shiqi Yu, all rights reserved.
#include "facedetectcnn.h"

"""
        self.convert()

    @staticmethod
    def combine_conv_bn(conv, bn):
        conv_result = nn.Conv2d(conv.in_channels, conv.out_channels, 
                                kernel_size=conv.kernel_size, stride=conv.stride, 
                                padding=conv.padding, groups = conv.groups, bias=True)
        scales = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        conv_result.bias.data = (conv.bias.data - bn.running_mean) * scales + bn.bias.data
        for ch in range(conv.out_channels):
                conv_result.weight.data[ch, ...] = conv.weight.data[ch, ...] * scales[ch]
        return conv_result

    @staticmethod
    def convert_param2string(conv, name, is_depthwise=False, withBNRelu=False, isfirst3x3x3=False, precision='.3g'):
        '''
        Convert the weights to strings
        '''
        (out_channels, in_channels, width, height) = conv.weight.size()

        if (isfirst3x3x3):
            w = conv.weight.detach().numpy().reshape((-1,27))
            _w = w.copy()
            for idx in range(out_channels):
                for offest in range(27):
                    w[idx, (offest % 9) * 3 + offest // 9] = _w[idx, offest]
            w_zeros = np.zeros((out_channels, 5))
            w = np.hstack((w, w_zeros))
            w = w.reshape(-1)
        elif (is_depthwise):
            w = conv.weight.detach().numpy().reshape((-1,9)).transpose().reshape(-1)
        else:
            w = conv.weight.detach().numpy().reshape(-1)

        b = conv.bias.detach().numpy().reshape(-1)

        data = {"type": "float", "weight_name": f"{name}_weight", "weight_size": "", "weight": "",\
            "bias_name": f"{name}_bias", "bias_size": "", "bias": "",\
            "with_bn": withBNRelu, "is_dw": is_depthwise, "in_channels": out_channels if is_depthwise else in_channels, "out_channels": out_channels}
        if (isfirst3x3x3):
            data["weight_size"] = str(out_channels) + '*32*1*1'
            data["in_channels"] = 32
        else:
            data["weight_size"] = str(out_channels) + '*' + str(in_channels) + '*' + str(width) + '*' + str(height)
        
        weight_str = ""
        for idx in range(w.size - 1):
            weight_str += (data2str_as_precision(w[idx], precision) + ',')

        weight_str += data2str_as_precision(w[-1], precision)
        data['weight'] = weight_str

        data["bias_size"] = str(out_channels)
        bias_str = ""
        for idx in range(b.size - 1):
            bias_str += (data2str_as_precision(b[idx], precision) + ',')
        bias_str += (data2str_as_precision(b[-1], precision))
        data['bias'] = bias_str

        return data


    def convert_module2string(self, conv, name, module_type):
        if module_type == "Conv_head":
            self.cppdata.append(self.convert_param2string(self.combine_conv_bn(conv.conv1, conv.bn1), name + '_pw', is_depthwise=False, withBNRelu=True, isfirst3x3x3=True))
            self.convert_module2string(conv.conv2, name + '_dp', module_type="ConvDPUnit")
        elif (module_type == "ConvDPUnit"):
            self.cppdata.append(self.convert_param2string(conv.conv1, name + '_pw', is_depthwise=False))
            if conv.withBNRelu:
                self.cppdata.append(self.convert_param2string(self.combine_conv_bn(conv.conv2, conv.bn), name + '_dw', is_depthwise=True, withBNRelu=True))
            else:
                self.cppdata.append(self.convert_param2string(conv.conv2, name + '_dw', is_depthwise=True))
        elif (module_type == "Conv4layerBlock"):
            self.convert_module2string(conv.conv1, name + "_dp1", "ConvDPUnit")
            self.convert_module2string(conv.conv2, name + "_dp2", "ConvDPUnit")
        else:
            raise ValueError(f"Unsupport module, please comfirm this module: {name} in {self.support_modules} !")


    def loop_search_modules(self, model, last_name=""):
        for name, m in model.named_children():
            t_name = f"{last_name}__{name}"
            m_str = m.__class__.__name__
            if(m_str in self.support_modules):
                self.module_list.append({"type": m_str, "name": t_name[2:], "module": m})
            else:
                self.loop_search_modules(m, t_name)
   
    @staticmethod
    def pythonBool2CBool(b):
        if(b):
            return "true"
        else:
            return "false"

    def convert(self):
        for m in self.module_list:
            self.convert_module2string(conv=m["module"], name=m["name"], module_type=m["type"])

        for d in self.cppdata:
            self.data += f"{d['type']} {d['weight_name']}[{d['weight_size']}] = {'{'}{d['weight']}{'}'};\n"
            self.data += f"{d['type']} {d['bias_name']}[{d['bias_size']}] = {'{'}{d['bias']}{'}'};\n"

        struct_str = "\n//(in_channels, out_channels, is_depthwise, is_pointwise, with_bn, weight_ptr, bias_ptr)\n"
        struct_str += f"ConvInfoStruct param_pConvInfo[{len(self.cppdata)}] = {'{'}\n"
        for i in range(len(self.cppdata) - 1):
            d = self.cppdata[i]
            struct_str += f"\t{'{'}{d['in_channels']}, {d['out_channels']}, {self.pythonBool2CBool(d['is_dw'])}"\
                + f", {self.pythonBool2CBool(not d['is_dw'])}, {self.pythonBool2CBool(d['with_bn'])}"\
                + f", {d['weight_name']}, {d['bias_name']}{'}'},\n"
        d = self.cppdata[-1]
        struct_str += f"\t{'{'}{d['in_channels']}, {d['out_channels']}, {self.pythonBool2CBool(d['is_dw'])}"\
            + f", {self.pythonBool2CBool(not d['is_dw'])}, {self.pythonBool2CBool(d['with_bn'])}"\
            + f", {d['weight_name']}, {d['bias_name']}{'}'}\n{'}'};"
        self.data += struct_str


if __name__ == "__main__":
    args = parse_args()
    model = build_model_from_cfg(args.config, args.checkpoint)

    if not args.no_summary:
        try:
            from mmcv.cnn import get_model_complexity_info
        except ImportError:
            raise ImportError('Please upgrade mmcv to >0.6.2')
        model.eval()
        if hasattr(model, 'forward_dummy'):
            model.forward = model.forward_dummy
        input_shape = (3, 320, 320)
        flops, params = get_model_complexity_info(model, input_shape, as_strings=False)
        split_line = '=' * 30
        print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')

    m_str = CppConvertor(model).data
    with open(args.output_file, "w") as f:
        f.write(m_str)
    print("Convert successful!")
    print(f"From {args.config} with {args.checkpoint}\nTo {args.output_file}")