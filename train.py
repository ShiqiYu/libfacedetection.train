import argparse
import os
import yaml
import time
import math
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from model import YuDetectNet

from tools import Logger, DaliWiderfaceDataset
# try:
#     import wandb
#     has_wandb = True
# except ImportError: 
#     has_wandb = False



parser = argparse.ArgumentParser(description='Yunet Training')
parser.add_argument('--config', '-c', default='./config/yufacedet.yaml', type=str, help='config to train')
parser.add_argument('--tag', '-t', default=None, type=str, help='tag to mark train process')
parser.add_argument('--resume', type=int, default=0, help='resume epoch to start')
def arg_initial(args):
    with open(args.config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    if not os.path.exists(cfg['train']['workspace']):
        os.mkdir(cfg['train']['workspace'])
        print(f"Create new folder (workspace): {os.path.abspath(cfg['train']['workspace'])}")
    if not cfg['train'].__contains__('tag'):
        if args.tag != None:
            cfg['train']['tag'] = cfg['train']['prefix'] + args.tag
            workfolder = os.path.join(cfg['train']['workspace'], cfg['train']['tag'])
        else:
            workfolder = os.path.join(cfg['train']['workspace'], 'debug')
            cfg['train']['tag'] = None
    else:
        workfolder = os.path.join(cfg['train']['workspace'], cfg['train']['tag'])

    if not os.path.exists(workfolder):
        os.mkdir(workfolder)
        print(f"Create new folder (workfolder): {os.path.abspath(workfolder)}")

    save_dir = os.path.join(workfolder, 'weights')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    cfg['train']['save_dir'] = save_dir
    log_dir = os.path.join(workfolder, 'log')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    cfg['train']['log_dir'] = log_dir

    if args.resume != 0:
        start_epoch = args.resume - args.resume % cfg['train']['save_interval']
        resume_weights = os.path.join(workfolder, 'weights', f"{cfg['train']['tag']}_epoch_{start_epoch}.pth")
        cfg['train']['resume_weights'] = resume_weights
        cfg['train']['start_epoch'] = start_epoch
    with open(os.path.join(workfolder, os.path.basename(args.config)), mode='w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f)

    return cfg

def log_initial(cfg):
    return Logger(cfg, mode='train')

def main():
    args = parser.parse_args()
    cfg = arg_initial(args)
    logger = log_initial(cfg)

    train_dataloader = DaliWiderfaceDataset(
                imgs_root=cfg['dataset']['img_root'],
                annos_file=cfg['dataset']['annotations_file'],
                batch_size=cfg['train']['batch_size'],
                num_workers=cfg['dataset']['num_workers'],
                device_id=0,
                local_seed=cfg['train']['seed'],
                shuffle=cfg['dataset']['shuffle'],
                shuffle_after_epoch=cfg['dataset']['shuffle_after_epoch'],
                num_gpus=1,
                img_dim=cfg['dataset']['image_size']
    )
    net = YuDetectNet(cfg)
    start_epoch = cfg['train']['start_epoch']
    if start_epoch != 0:
        logger.info(f"Resume training process: {cfg['train']['resume_weights']}")
        logger.epoch = start_epoch - 1
        net.load_state_dict(torch.load(cfg['train']['resume_weights']))
    net = net.cuda()
    cudnn.benchmark = True
    optimizer = optim.SGD(
        net.parameters(), 
        lr=cfg['train']['lr'], 
        momentum=cfg['train']['momentum'], 
        weight_decay=cfg['train']['weight_decay']
    )

    for epoch in range(start_epoch, cfg['train']['max_epochs']):
        lr = adjust_learning_rate_poly(optimizer, cfg['train']['lr'], epoch, cfg['train']['max_epochs'])
        epoch_time = train_one_epoch(net, train_dataloader, optimizer, lr, logger)

        # multi_scale training
        train_dataloader.reset()

        logger.info('Epoch time: {:.2f} minutes; Time left: {:.2f} hours'.format(epoch_time/60, \
                    (epoch_time)*(cfg['train']['max_epochs']-epoch-1)/3600))
        if ((epoch + 1) % cfg['train']['save_interval'] == 0) :
            save_path = os.path.join(cfg['train']['save_dir'], \
                                    f"{cfg['train']['tag']}_epoch_{epoch + 1}.pth")
            torch.save(net.state_dict(), save_path)
            logger.info(f'Save state_dict to: {save_path}')

    torch.save(net.state_dict(), os.path.join(cfg['train']['save_dir'], f"{cfg['train']['tag']}_final.pth"))
    

def train_one_epoch(net, train_dataloader, optimizer, lr, logger):
    max_iter = len(train_dataloader)
    epoch_t0 = time.time()
    for iter_idx, one_batch_data in enumerate(train_dataloader):
        images, targets = one_batch_data
        images = images.cuda()
        targets = [anno.cuda() for anno in targets]
        _, _, h, w = images.shape
        pred = net(images)
        loss_bbox_eiou, loss_iouhead_smoothl1, loss_lm_smoothl1, loss_cls_ce = net.loss(pred, targets)
        loss = loss_bbox_eiou + loss_iouhead_smoothl1 + loss_lm_smoothl1 + loss_cls_ce
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        msg = {
            'loss_bbox': loss_bbox_eiou.item(),
            'loss_iouhead': loss_iouhead_smoothl1.item(),
            'loss_lmds': loss_lm_smoothl1.item(),
            'loss_cls': loss_cls_ce.item(),
            'loss_all': loss.item(),
            'lr': lr,
            'iter': iter_idx,
            'max_iter': max_iter,
            'tag': {'img_size': (h, w)}
        }
        logger.update(msg)
    return time.time() - epoch_t0

def adjust_learning_rate_poly(optimizer, initial_lr, iteration, max_iter, mode='poly'):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    if mode == 'poly':
        lr = initial_lr * ( 1 - (iteration / max_iter)) * ( 1 - (iteration / max_iter))
        if ( lr < 1.0e-7 ):
            lr = 1.0e-7
    elif mode == 'cos':
        lr_min = 1.0e-7
        T_max = 100
        lr = (lr_min + (initial_lr - lr_min) * (1 + math.cos(math.pi * iteration / T_max)) * 0.5) * ( 1 - (iteration / max_iter))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr
        

if __name__ == "__main__":
    main()