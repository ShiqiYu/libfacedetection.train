#!/usr/bin/python3
from __future__ import print_function

import os
import sys
import torch
import torch.backends.cudnn as cudnn
import argparse
import cv2
import numpy as np
from collections import OrderedDict

sys.path.append(os.getcwd() + '/../../src')

from config import cfg
from prior_box import PriorBox
from nms import nms
from utils import decode
from timer import Timer
from yufacedetectnet import YuFaceDetectNet

def str2bool(v):
    if v.lower() in ['on', 'yes', 'true']:
        return True
    elif v.lower() in ['off', 'no', 'false']:
        return False
    else:
        raise NotImplementedError

parser = argparse.ArgumentParser(description='Face and Landmark Detection')

parser.add_argument('-m', '--trained_model', default='weights/yunet_final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('-o', '--output_name', default='yunet',
                    type=str, help='The output ONNX file, trained parameters inside')
parser.add_argument('--enable_dynamic_axes', default=True,
                    type=str2bool, help='Enable dynamic axes for ONNX model.')
parser.add_argument('--opset_version', default=11, help='ONNX opset version to output.')
args = parser.parse_args()

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


if __name__ == '__main__':

    torch.set_grad_enabled(False)

    # net and model
    net = YuFaceDetectNet(phase='test', size=None )    # initialize detector
    net = load_model(net, args.trained_model, True)
    net.eval()

    print('Finished loading model!')

    img = torch.randn(1, 3, 480, 640, requires_grad=False)
    img = img.to(torch.device('cpu'))

    input_names = ['input']
    output_names = ['loc', 'conf', 'iou']
    if args.enable_dynamic_axes:
        dynamic_axes = {'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                        'loc':   {0: 'batch_size', 1: 'num', 2: 'loc_data'},
                        'conf':  {0: 'batch_size', 1: 'num', 2: 'cls_data'},
                        'iou':   {0: 'batch_size', 1: 'num', 2: 'iou_data'}}
        output_path = os.path.join('./onnx', args.output_name + '.onnx')
        torch.onnx.export(net, img, output_path, input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes, opset_version=args.opset_version)
    else:
        output_path = os.path.join('./onnx', args.output_name + '_' + str(args.image_dim) + '.onnx')
        torch.onnx.export(net, img, output_path, input_names=input_names, output_names=output_names, opset_version=args.opset_version)
    print('Finished exporing model to ' + output_path)
