import argparse
import os

import mmcv
import torch
from auto_rank_result import AutoRank
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from mmdet.core.evaluation import wider_evaluation
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--out', default='./work_dirs/wout', help='output folder')
    parser.add_argument(
        '--save-preds', action='store_true', help='save results')

    parser.add_argument(
        '--thr', type=float, default=-1., help='score threshold')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--mode',
        type=int,
        default=0,
        help="""
            mode    test resolution
            0       (640, 640)
            1       (1100, 1650)
            2       Origin Size diveisor=32
            >30     (mode, mode)
            """)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

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

    gt_path = os.path.join(os.path.dirname(cfg.data.test.ann_file), 'gt')
    pipelines = cfg.data.test.pipeline
    for pipeline in pipelines:
        if pipeline.type == 'MultiScaleFlipAug':
            if args.mode == 0:  # 640 scale
                pipeline.img_scale = (640, 640)
            elif args.mode == 1:  # for single scale in other pages
                pipeline.img_scale = (1100, 1650)
            elif args.mode == 2:  # original scale
                pipeline.img_scale = None
                pipeline.scale_factor = 1.0
            elif args.mode > 30:
                pipeline.img_scale = (args.mode, args.mode)
            transforms = pipeline.transforms
            for transform in transforms:
                if transform.type == 'Pad':
                    if args.mode != 2:
                        transform.size = pipeline.img_scale
                    else:
                        transform.size = None
                        transform.size_divisor = 32
    print(cfg.data.test.pipeline)
    distributed = False

    # build the dataloader
    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
    if samples_per_gpu > 1:
        # Replace 'ImageToTensor' to 'DefaultFormatBundle'
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)
    if args.thr != -1.:
        cfg.model.test_cfg.score_thr = args.thr

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=None)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    results = {}
    output_folder = args.out
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        assert len(result) == 1
        batch_size = 1
        result = result[0][0]
        img_metas = data['img_metas'][0].data[0][0]
        filepath = img_metas['ori_filename']
        # det_scale = img_metas['scale_factor'][0]

        _vec = filepath.split('/')
        pa, pb = _vec[-2], _vec[1]
        if pa not in results:
            results[pa] = {}
        xywh = result.copy()
        w = xywh[:, 2] - xywh[:, 0]
        h = xywh[:, 3] - xywh[:, 1]
        xywh[:, 2] = w
        xywh[:, 3] = h

        event_name = pa
        img_name = pb.rstrip('.jpg')
        results[event_name][img_name] = xywh
        if args.save_preds:
            out_dir = os.path.join(output_folder, pa)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            out_file = os.path.join(out_dir, pb.replace('jpg', 'txt'))
            boxes = result
            with open(out_file, 'w') as f:
                name = '/'.join([pa, pb])
                f.write('%s\n' % (name))
                f.write('%d\n' % (boxes.shape[0]))
                for b in range(boxes.shape[0]):
                    box = boxes[b]
                    f.write('%.5f %.5f %.5f %.5f %g\n' %
                            (box[0], box[1], box[2] - box[0], box[3] - box[1],
                             box[4]))

        for _ in range(batch_size):
            prog_bar.update()
    aps = wider_evaluation(results, gt_path, 0.5)

    AutoRank('./eval.log').update({
        'config':
        args.config,
        'weight':
        args.checkpoint,
        'score_nms_thresh':
        [cfg.model.test_cfg.score_thr, cfg.model.test_cfg.nms.iou_threshold],
        'APS':
        aps
    })

    with open(os.path.join(output_folder, 'aps'), 'w') as f:
        f.write('%f,%f,%f\n' % (aps[0], aps[1], aps[2]))
    print('APS:', aps)


if __name__ == '__main__':
    main()
