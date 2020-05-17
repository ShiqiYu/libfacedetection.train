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

parser = argparse.ArgumentParser(description='Face and Landmark Detection')

parser.add_argument('-m', '--trained_model', default='weights/yunet_final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('-d', '--image_dim', default=320,
                    type=int, help='Input image width')
parser.add_argument('-o', '--output', default='onnx/facedetectcnn.onnx',
                    type=str, help='The output ONNX file, trained parameters inside')
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
    
    height = 0.75 * args.image_dim
    img_raw = np.zeros((args.image_dim, int(height), 3), np.uint8)
    img = np.float32(img_raw)

    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(torch.device('cpu'))
    torch.onnx.export(net, img, args.output)
    print('Finished exporing model to ' + args.output)
