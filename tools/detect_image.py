import argparse
from pyexpat import model
import numpy as np

import torch
from mmcv import Config
from mmcv.runner import (load_checkpoint)

from mmdet.models import build_detector

import cv2



def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('image', help="input image")
    parser.add_argument(
        '--thr',
        type=float,
        default=-1.,
        help='score threshold')
    parser.add_argument('--out', type=str, default="./result.jpg")    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()


    cfg = Config.fromfile(args.config)
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True

    model = build_detector(cfg.model, train_cfg=None, test_cfg=None)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.eval()

    image = cv2.imread(args.image)
    # ori_h, ori_w, _ = image.shape
    # image = cv2.resize(image, (640, 640))
    # cv2.imwrite("./test.jpg", image)
    image_tensor = torch.from_numpy(image).float()
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        # result = model(return_loss=False, rescale=False, **data) 
        result = model.simple_test(image_tensor, None)         
    assert len(result)==1
    result = result[0][0]
    draw(image, result, None, args.out, True)


def draw(img, bboxes, kpss, out_path, with_kps=True):
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i]
        x1,y1,x2,y2,score = bbox.astype(np.int32)
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255) , 2)
        if with_kps:
            if kpss is not None:
                kps = kpss[i].reshape(-1, 2)
                for kp in kps:
                    kp = kp.astype(np.int32)
                    cv2.circle(img, tuple(kp) , 1, (255,0,0) , 2)
        
    print('output:', out_path)
    cv2.imwrite(out_path, img) 


if __name__ == '__main__':
    main()