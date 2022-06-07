import argparse
import os
import yaml
import glob
import torch
import torch.backends.cudnn as cudnn
import cv2
import time
import numpy as np

from tools import Logger
from model import YuDetectNet
import glob

parser = argparse.ArgumentParser(description='Face and Landmark Detection')
parser.add_argument('--config', '-c', type=str, help='config to test')
parser.add_argument('--model', '-m', type=str, help='model path to test')
parser.add_argument('-t', '--target', type=str, help='image/image folder/video path')
parser.add_argument('--confidence_threshold', type=float, help='confidence threshold to save result')
parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda:0')

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
    save_dir = os.path.join(workfolder, 'detect_results')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    cfg['test']['save_dir'] = save_dir 
    assert os.path.exists(args.target)

    if args.confidence_threshold is not None:
        cfg['test']['confidence_threshold'] = args.confidence_threshold
        
    return cfg


def detect_image(net, img_path, cfg, device):
    img = cv2.imread(img_path)

    t0 = time.time()
    dets = net.inference(img, scale=1., without_landmarks=False, device=device)
    t1 = time.time()

    dets = dets.cpu().numpy()
    if len(dets) == 0:
        print('Detect 0 taeget!')
        return t1 - t0
    scores = dets[:, -1]
    dets = dets[:, :-1].astype(np.int32)
    for det, score in zip(dets, scores):
        img = cv2.rectangle(img, (det[0], det[1]), (det[2], det[3]), color=(0, 0, 255), thickness=1)
        # img = cv2.putText(img, f"{score:4f}", (det[0], det[1] + 12), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
        ldms_num = int((det.shape[-1] - 4) / 2)
        for i in range(ldms_num):
            cv2.circle(img, (det[4 + 2 * i], det[5 + 2 * i]), 2, (255, 255, 0), thickness=1)
    save_path = os.path.join(cfg['test']['save_dir'], os.path.basename(img_path))
    cv2.imwrite(save_path, img)
    print(f'Detect {0 if len(dets.shape) == 1 else dets.shape[0]} target, Save img to {save_path}')
    return t1 - t0

def detect_video(net, video_path, cfg, device):
    cap = cv2.VideoCapture(video_path)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    save_path = os.path.join(cfg['test']['save_dir'], os.path.basename(video_path))
    video_writer = cv2.VideoWriter(
                save_path, 
                cv2.VideoWriter_fourcc('M','P','E','G'),
                fps,
                size
    )
    print('Detect video, please wait ...')
    total_time = 0
    num_frames = 0
    while(True):
        ret, frame = cap.read()
        if ret:
            t0 = time.time()
            det = net.inference(frame, scale=1., without_landmarks=False, device=device)
            total_time += time.time() - t0
            num_frames += 1
            det = det.cpu().numpy()            
            scores = det[:, -1]
            det = det[:, :-1].astype(np.int32)
            for det, score in zip(det, scores):
                frame = cv2.rectangle(frame, (det[0], det[1]), (det[2], det[3]), color=(0, 0, 255), thickness=1)
                frame = cv2.putText(frame, f"{score:4f}", (det[0], det[1] + 12), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                for i in range(int((det.shape[-1] - 4) / 2)):
                    cv2.circle(frame, (det[4 + 2 * i], det[5 + 2 * i]), 2, (255, 255, 0), thickness=5)
            video_writer.write(frame)
        else:
            break
    cap.release()
    video_writer.release()
    print(f'Save video to {save_path}')
    return total_time, num_frames


def main():
    args = parser.parse_args()
    cfg = arg_initial(args)
    torch.set_grad_enabled(False)

    print(f'Loading model from {args.model}')
    net = YuDetectNet(cfg)
    net.load_state_dict(torch.load(args.model))
    net.eval()
    net.to(args.device)
    cudnn.benchmark = True

    target = args.target
    img_paths = []
    if os.path.isdir(target):
        img_paths.extend(glob.glob(os.path.join(target, '*.jpg')))
        img_paths.extend(glob.glob(os.path.join(target, '*.jpeg'))) 
        img_paths.extend(glob.glob(os.path.join(target, '*.png')))         
        img_paths.extend(glob.glob(os.path.join(target, '*.mp4')))
    else:
        img_paths.append(target)

    print(f'{len(img_paths)} files to be detected...')

    total_time = 0
    num_frames = 0
    for img_path in img_paths:
        filename, tp = os.path.splitext(os.path.basename(img_path))
        if tp.lower() in ('.jpg', '.jpeg', '.png'):
            total_time += detect_image(net, img_path, cfg, device=args.device)
            num_frames += 1
        elif tp.lower() in ('.mp4'):
            _total_time, _num_frames = detect_video(net, img_path, cfg, device=args.device)
            total_time += _total_time
            num_frames += _num_frames
        else:
            print(f'{img_path}: Unsupport file!')
    
    print(f'Detect {num_frames} images and achieve {int(num_frames / total_time)} fps.')



if __name__ == "__main__":
    main()