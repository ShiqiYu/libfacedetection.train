#!/usr/bin/python3
from __future__ import print_function

import os
import sys
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data

import argparse
import time
import datetime
import math
import numpy as np

from config import cfg

sys.path.append(os.getcwd() + '/../../src')

from data import FaceRectLMDataset, detection_collate
from multibox_loss import MultiBoxLoss
from prior_box import PriorBox
from yufacedetectnet import YuFaceDetectNet

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20000, rlimit[1]))

parser = argparse.ArgumentParser(description='YuMobileNet Training')
parser.add_argument('--training_face_rect_dir', default='../../data/WIDER_FACE_rect', help='Training dataset directory')
parser.add_argument('--training_face_landmark_dir', default='../../data/WIDER_FACE_landmark', help='Training dataset directory')
parser.add_argument('-b', '--batch_size', default=16, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
parser.add_argument('--gpu_ids', default='0', help='the IDs of GPU')
parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('-max', '--max_epoch', default=500, type=int, help='max epoch for retraining')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--weight_filename_prefix', default='weights/yunet', help='the prefix of the weight filename')
args = parser.parse_args()


img_dim = 320 # only 1024 is supported
rgb_mean =  (0,0,0) #(104, 117, 123) # bgr order
num_classes = 2
gpu_ids =  [int(item) for item in args.gpu_ids.split(',')]
num_workers = args.num_workers
#batch_size = args.batch_size
momentum = args.momentum
weight_decay = args.weight_decay
#initial_lr = args.lr
gamma = args.gamma
max_epoch = args.max_epoch
training_face_rect_dir = args.training_face_rect_dir
training_face_landmark_dir = args.training_face_landmark_dir

net = YuFaceDetectNet('train', img_dim)
print("Printing net...")
#print(net)

if args.resume_net is not None:
    print('Loading resume network...')
    state_dict = torch.load(args.resume_net)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

if len(gpu_ids) > 1 :
    net = torch.nn.DataParallel(net, device_ids=gpu_ids)

#device = torch.device(args.device)
device = torch.device('cuda:'+str(gpu_ids[0]))
cudnn.benchmark = True
net = net.to(device)

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=momentum, weight_decay=weight_decay)
criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 3, 0.35, False, False)

priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
with torch.no_grad():
    priors = priorbox.forward()
    priors = priors.to(device)



def train():

    net.train()

    #load the two dataset for face rectangles and landmarks respectively
    print('Loading Dataset...')
    dataset_rect = FaceRectLMDataset(training_face_rect_dir, img_dim, rgb_mean)
    dataset_landmark = FaceRectLMDataset(training_face_landmark_dir, img_dim, rgb_mean)
    
    batch_size = args.batch_size

    for epoch in range(args.resume_epoch, max_epoch):
        if epoch < 100 :
            with_landmark = False
        else:
            with_landmark = (epoch % 2 == 1)

        dataset = dataset_rect
        if with_landmark:
            dataset = dataset_landmark

        train_loader = data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=detection_collate,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=True,
        )

        lr = adjust_learning_rate_poly(optimizer, args.lr, epoch, max_epoch)

        #for computing average losses in this epoch
        loss_l_epoch = []
        loss_lm_epoch = []
        loss_c_epoch = []
        loss_epoch = []

        # the start time
        load_t0 = time.time()

        # for each iteration in this epoch
        num_iter_in_epoch = len(train_loader)
        for iter_idx, one_batch_data in enumerate(train_loader):
            # load train data
            #images, targets = next(batch_iterator)
            images, targets = one_batch_data
            images = images.to(device)
            targets = [anno.to(device) for anno in targets]

            # forward
            out = net(images)
            # loss
            loss_l, loss_lm, loss_c = criterion(out, priors, targets)

            if with_landmark:
                loss = loss_l + loss_lm + loss_c
            else:
                loss = loss_l + loss_c

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # put losses to lists to average for printing
            loss_l_epoch.append(loss_l.item())
            loss_lm_epoch.append(loss_lm.item())
            loss_c_epoch.append(loss_c.item())
            loss_epoch.append(loss.item())

            # print loss
            if ( iter_idx % 20 == 0 or iter_idx == num_iter_in_epoch - 1):
                print('LM:{} || Epoch:{}/{} || iter: {}/{} || L: {:.2f}({:.2f}) LM: {:.2f}({:.2f}) C: {:.2f}({:.2f}) All: {:.2f}({:.2f}) || LR: {:.8f}'.format(
                    with_landmark, epoch, max_epoch, iter_idx, num_iter_in_epoch, 
                    loss_l.item(), np.mean(loss_l_epoch), 
                    loss_lm.item(), np.mean(loss_lm_epoch), 
                    loss_c.item(), np.mean(loss_c_epoch), 
                    loss.item(),  np.mean(loss_epoch), lr))


        if (epoch % 50 == 0 and epoch > 0) :
            torch.save(net.state_dict(), args.weight_filename_prefix + '_epoch_' + str(epoch) + '.pth')

        #the end time
        load_t1 = time.time()
        epoch_time = (load_t1 - load_t0) / 60
        print('Epoch time: {:.2f} minutes; Time left: {:.2f} hours'.format(epoch_time, (epoch_time)*(max_epoch-epoch-1)/60 ))

    torch.save(net.state_dict(), args.weight_filename_prefix + '_final.pth')


def adjust_learning_rate_poly(optimizer, initial_lr, iteration, max_iter):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = initial_lr * ( 1 - (iteration / max_iter)) * ( 1 - (iteration / max_iter))
    if ( lr < 1.0e-7 ):
      lr = 1.0e-7

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr;

    return lr


if __name__ == '__main__':
    train()
