#!/usr/bin/python3
from __future__ import print_function

import os
import sys
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

import argparse
import time
import numpy as np

from config import cfg

sys.path.append(os.getcwd() + '/../../src')

from data import get_train_loader
from multibox_loss import MultiBoxLoss
from prior_box import PriorBox
from yufacedetectnet import YuFaceDetectNet

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20000, rlimit[1]))

def str2bool(s):
    if s.lower() in ['true', 'yes', 'on']:
        return True
    return False

parser = argparse.ArgumentParser(description='YuMobileNet Training')
parser.add_argument('--dataset_dir', default='../../data/widerface', help='dataset directory')
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
parser.add_argument('--lambda_bbox_eiou', default=10, type=int, help='lambda for bbox reg loss')
parser.add_argument('--lambda_iouhead_smoothl1', default=1, type=int, help='lambda for iou head loss')
parser.add_argument('--lambda_lm_smoothl1', default=1, type=float, help='lambda for landmark reg loss')
parser.add_argument('--lambda_cls_ce', default=1, type=int, help='lambda for classification loss')
parser.add_argument('--use_tensorboard', default=True, type=str2bool, help='True to use tensorboard.')
args = parser.parse_args()



if args.use_tensorboard:
    from torch.utils.tensorboard import SummaryWriter
    prefix = args.weight_filename_prefix.split('/')[-1]
    log_tag = '{prefix}-bbox_{l_bbox}-iouhead_{l_iouhead}-lm_{l_lm}-cls_{l_cls}'.format(
        prefix=prefix,
        l_bbox=args.lambda_bbox_eiou,
        l_iouhead=args.lambda_iouhead_smoothl1,
        l_lm=args.lambda_lm_smoothl1,
        l_cls=args.lambda_cls_ce
    )
    logger = SummaryWriter(os.path.join('./tb_logs', log_tag))

img_dim = 320 # only 1024 is supported
num_classes = 2
gpu_ids =  [int(item) for item in args.gpu_ids.split(',')]
num_workers = args.num_workers
#batch_size = args.batch_size
momentum = args.momentum
weight_decay = args.weight_decay
#initial_lr = args.lr
gamma = args.gamma
max_epoch = args.max_epoch
dataset_dir = args.dataset_dir

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
    batch_size = args.batch_size

    train_loader = get_train_loader(
        imgs_root=os.path.join(args.dataset_dir, 'WIDER_train/images'),
        annos_file=os.path.join(args.dataset_dir,'trainset.json'),
        batch_size=batch_size,
        num_workers=num_workers,
        device_id=0,
        local_seed=-1,
        shuffle=True,
        shuffle_after_epoch=False,
        num_gpus=1,
    )

    for epoch in range(args.resume_epoch, max_epoch):
        lr = adjust_learning_rate_poly(optimizer, args.lr, epoch, max_epoch)

        #for computing average losses in this epoch
        loss_bbox_epoch = []
        loss_iouhead_epoch = []
        loss_lm_epoch = []
        loss_cls_epoch = []
        loss_epoch = []

        # the start time
        load_t0 = time.time()

        # for each iteration in this epoch
        num_iter_in_epoch = len(train_loader)
        for iter_idx, one_batch_data in enumerate(train_loader):
            # load train data
            images, targets = one_batch_data
            images = images.to(device)
            targets = [anno.to(device) for anno in targets]

            # forward
            out = net(images)
            # loss
            loss_bbox_eiou, loss_iouhead_smoothl1, loss_lm_smoothl1, loss_cls_ce = criterion(out, priors, targets)

            loss = args.lambda_bbox_eiou * loss_bbox_eiou + \
                   args.lambda_iouhead_smoothl1 * loss_iouhead_smoothl1 + \
                   args.lambda_lm_smoothl1 * loss_lm_smoothl1 + \
                   args.lambda_cls_ce * loss_cls_ce

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # put losses to lists to average for printing
            loss_bbox_epoch.append(loss_bbox_eiou.item())
            loss_iouhead_epoch.append(loss_iouhead_smoothl1.item())
            loss_lm_epoch.append(loss_lm_smoothl1.item())
            loss_cls_epoch.append(loss_cls_ce.item())
            loss_epoch.append(loss.item())

            if args.use_tensorboard:
                logger.add_scalar(
                    tag='Iter/loss_bbox',
                    scalar_value=loss_bbox_eiou.item(),
                    global_step=iter_idx+epoch*num_iter_in_epoch
                )
                logger.add_scalar(
                    tag='Iter/loss_iou',
                    scalar_value=loss_iouhead_smoothl1.item(),
                    global_step=iter_idx+epoch*num_iter_in_epoch
                )
                logger.add_scalar(
                    tag='Iter/loss_landmark',
                    scalar_value=loss_lm_smoothl1.item(),
                    global_step=iter_idx+epoch*num_iter_in_epoch
                )
                logger.add_scalar(
                    tag='Iter/loss_cls',
                    scalar_value=loss_cls_ce.item(),
                    global_step=iter_idx+epoch*num_iter_in_epoch
                )

            # print loss
            if (iter_idx % 20 == 0 or iter_idx == num_iter_in_epoch - 1):
                print('Epoch:{}/{} || iter: {}/{} || L: {:.2f}({:.2f}) IOU: {:.2f}({:.2f}) LM: {:.2f}({:.2f}) C: {:.2f}({:.2f}) All: {:.2f}({:.2f}) || LR: {:.8f}'.format(
                    epoch, max_epoch, iter_idx, num_iter_in_epoch, 
                    loss_bbox_eiou.item(), np.mean(loss_bbox_epoch),
                    loss_iouhead_smoothl1.item(), np.mean(loss_iouhead_epoch),
                    loss_lm_smoothl1.item(), np.mean(loss_lm_epoch),
                    loss_cls_ce.item(), np.mean(loss_cls_epoch),
                    loss.item(),  np.mean(loss_epoch), lr))

        if args.use_tensorboard:
            logger.add_scalar(
                tag='Epoch/loss_bbox',
                scalar_value=np.mean(loss_bbox_epoch),
                global_step=epoch
            )
            logger.add_scalar(
                tag='Epoch/loss_iouhead',
                scalar_value=np.mean(loss_iouhead_epoch),
                global_step=epoch
            )
            logger.add_scalar(
                tag='Epoch/loss_landmark',
                scalar_value=np.mean(loss_lm_epoch),
                global_step=epoch
            )
            logger.add_scalar(
                tag='Epoch/loss_cls',
                scalar_value=np.mean(loss_cls_epoch),
                global_step=epoch
            )

        if (epoch % 50 == 0 and epoch > 0) :
            torch.save(net.state_dict(), args.weight_filename_prefix + '_epoch_' + str(epoch) + '.pth')

        #the end time
        load_t1 = time.time()
        epoch_time = (load_t1 - load_t0) / 60
        print('Epoch time: {:.2f} minutes; Time left: {:.2f} hours'.format(epoch_time, (epoch_time)*(max_epoch-epoch-1)/60))

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
        param_group['lr'] = lr

    return lr


if __name__ == '__main__':
    train()
