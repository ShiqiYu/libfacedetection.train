import argparse
import os
import yaml
import glob
from tqdm import tqdm
import sys
import torch
import torch.backends.cudnn as cudnn
import cv2
import time
import numpy as np

from tools import Logger
from model import YuDetectNet
from tools import get_testloader, evaluation

parser = argparse.ArgumentParser(description='Face and Landmark Detection')
parser.add_argument('--config', '-c', type=str, help='config to test')
parser.add_argument('--model', '-m', type=str, help='model path to test')
parser.add_argument('--confidence_threshold', type=float, help='confidence threshold to save result')
parser.add_argument('--device', '-d', type=str, default='cuda:0', help='device to inference, cpu or cuda:0')
parser.add_argument('--multi-scale', action="store_true", help="multi-scale test")

def arg_initial(args):
    if args.config is not None:
        assert os.path.exists(args.config)
        workfolder = os.path.join("./workspace", os.path.basename(args.config)[:-5])
        if not os.path.exists(workfolder):
            os.makedirs(workfolder)
    else:
        workfolder = os.path.dirname(os.path.dirname(args.model))
        cfg_list = glob.glob(os.path.join(workfolder, '*.yaml'))
        assert len(cfg_list) == 1, 'Can`t comfire config file!'
        args.config = cfg_list[0]
    with open(args.config, mode='r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    if args.confidence_threshold is not None:
        cfg['test']['confidence_threshold'] = args.confidence_threshold

    log_dir = os.path.join(workfolder, 'log')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    cfg['test']['log_dir'] = log_dir
    save_dir = os.path.join(workfolder, 'results')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        print(f"Create new folder (save folder): {os.path.abspath(save_dir)}")
    cfg['test']['save_dir'] = save_dir    
    # with open(os.path.join(workfolder, os.path.basename(args.config)), mode='w', encoding='utf-8') as f:
    #     yaml.safe_dump(cfg, f)
    return cfg

def log_initial(cfg):
    return Logger(cfg, mode='test')

def main():
    args = parser.parse_args()
    cfg = arg_initial(args)
    logger = log_initial(cfg)
    torch.set_grad_enabled(False)

    logger.info(f'Loading model from {args.model}')
    net = YuDetectNet(cfg)
    net.load_state_dict(torch.load(args.model), strict=True)
    net.eval()
    net.to(args.device)
    cudnn.benchmark = True
    testloader = get_testloader(
                mode=cfg['test']['dataset']['mode'],
                split= cfg['test']['dataset']['split'],
                root=cfg['test']['dataset']['root']
    )
    scales = [0.25, 0.50, 0.75, 1.25, 1.50, 1.75, 2.0] if cfg['test']['multi_scale'] or args.multi_scale else [1.]
    logger.info(f'Performing testing with scales: {str(scales)}, conf_threshold: {cfg["test"]["confidence_threshold"]}')
    results = []
    for idx in tqdm(range(len(testloader))):
    # for idx in range(len(widerface)):
        img, mata = testloader[idx] # img_subpath = '0--Parade/XXX.jpg'
        available_scales = get_available_scales(img.shape[0], img.shape[1], scales)
        # available_scales = [1.2]
        dets = torch.empty((0, 5)).to(args.device)
        for available_scale in available_scales:
            det = net.inference(img, available_scale, device=args.device)
            dets = torch.cat([dets, det[:, [0,1,2,3,-1]]], dim=0)
        results.append({'pred': dets.cpu().numpy(), 'mata': mata})

    logger.info('Evaluating:')
    sys.stdout = logger
    evaluation(
                mode=cfg['test']['dataset']['mode'],
                results=results,
                results_save_dir=cfg['test']['save_dir'],
                gt_root=os.path.join(cfg['test']['dataset']['root'], 'ground_truth'),
                iou_tresh=cfg['test']['ap_threshold'],
                split= cfg['test']['dataset']['split']
    )

def draw(img, pred, idx = 0):
    scores = pred[:, -1]
    dets = pred[:, :-1].astype(np.int32)
    for det, score in zip(dets, scores):
        img = cv2.rectangle(img, (det[0], det[1]), (det[2], det[3]), color=(0, 0, 255), thickness=1)
        # img = cv2.putText(img, f"{score:4f}", (det[0], det[1] + 12), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
        # for i in range(4):
        #     cv2.circle(img, (det[4 + 2 * i], det[5 + 2 * i]), 2, (255, 255, 0), thickness=5)
    save_dir = "./results"
    cv2.imwrite( os.path.join(save_dir, f"{idx}_{score:.4f}.jpg"), img)

def get_available_scales(h, w, scales):
    smin = min(h, w)
    available_scales = []
    for scale in scales:
        if int(smin * scale) >= 64:
            available_scales.append(scale)
    return available_scales

if __name__ == "__main__":
    main()
