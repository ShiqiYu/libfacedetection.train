from .data import DaliWiderfaceDataset, WIDERFace, HanCotestloader, CCPDtestloader
from .log import Logger
from .widerface_eval.evaluation import evaluation as widerface_evaluation

import os
import numpy as np
import pickle

TEST_MODE = ('WIDERFACE', 'HANCO', 'CCPD')
def get_testloader(mode :str, **kargs):
    mode = mode.upper()
    assert mode in TEST_MODE, f'Test mode must be one of {TEST_MODE}'
    loader = {}
    loader['WIDERFACE'] = WIDERFace
    loader['HANCO'] = HanCotestloader
    loader['CCPD'] = CCPDtestloader
    return loader[mode](kargs)

def evaluation(mode, **kargs):
    mode = mode.upper()
    assert mode in TEST_MODE, f'Test mode must be one of {TEST_MODE}'
    eval_fn = {}
    eval_fn['WIDERFACE'] = eval_widerface
    eval_fn['HANCO'] = eval_hanco
    eval_fn['CCPD'] = eval_ccpd

    return eval_fn[mode](kargs)

def eval_widerface(kargs: dict):
    results = kargs.get('results', None)

    gt_path = kargs.get('gt_root', None)
    iou_thresh = kargs.get('iou_tresh', 0.5)
    save_path = kargs.get('results_save_dir', None)
    for result in results:
        event = result['mata']['event']
        name = result['mata']['name']
        preds = result['pred']
        save_res(preds, event, name, save_path)
    widerface_evaluation(
        pred=save_path,
        gt_path=gt_path,
        iou_thresh=iou_thresh
    )

def save_res(dets, event, name, save_path):
    txt_name = name[:-4]+'.txt'
    save_path = os.path.join(save_path, event)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(os.path.join(save_path, txt_name), 'w') as f:
        f.write('{}\n'.format('/'.join([event, name])))
        f.write('{}\n'.format(dets.shape[0]))
        for k in range(dets.shape[0]):
            xmin = dets[k, 0]
            ymin = dets[k, 1]
            xmax = dets[k, 2]
            ymax = dets[k, 3]
            score = dets[k, 4]
            w = xmax - xmin + 1
            h = ymax - ymin + 1
            f.write(f'{np.floor(xmin):.1f} {np.floor(ymin):.1f} {np.ceil(w):.1f} {np.ceil(h):.1f} {score:.3f}\n')
    
def eval_hanco(kargs: dict):
    results = kargs.get('results', None)
    results_save_path = os.path.join(kargs.get('results_save_dir'), 'result.pkl')
    split = kargs.get('split', None)
    gt_root=kargs.get('gt_root', None)

    preds = np.empty((0, 7))
    for result in results:
        det = result['pred']
        pred = np.ones((det.shape[0], 7), dtype=np.float32)
        pred[:, 0] *= float(result['mata']['id'])
        pred[:, 1:-1] = det
        preds = np.concatenate([preds, pred], axis=0)

    print(f'Save result to {results_save_path}')
    with open(results_save_path, 'wb') as f:
        pickle.dump(preds, f)

    print('COCO evaluation ...')
    try:
        from .coco_eval.PythonAPI.pycocotools import COCO, COCOeval
    except ImportError:
        print('The pycocotools API can not find, \nIf you are evaling, you should run ./tools/coco_eval/make_coco.sh.')
    gt_path = os.path.join(gt_root, 'detection_merge',f"{split}.json")
    coco_gt = COCO(gt_path)
    coco_dt = coco_gt.loadRes(results)
    cocoEval = COCOeval(coco_gt, coco_dt, "bbox")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

def eval_ccpd(kargs: dict):
    pass