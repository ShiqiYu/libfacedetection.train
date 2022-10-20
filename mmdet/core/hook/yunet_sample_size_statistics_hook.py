import json
import os
from datetime import datetime

from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class YuNetSampleSizeStatisticsHook(Hook):

    def __init__(self, out_file, save_interval=50) -> None:
        super().__init__()
        self.size_container = {}
        self.shapeless2 = 0
        self.noimg = 0
        self.out_file = out_file
        self.total_sample_num = 0
        self.save_interval = save_interval
        self.batch_size = 0

    def before_run(self, runner):
        work_dir = runner.work_dir
        self.out_file = os.path.join(work_dir, self.out_file)

    def before_epoch(self, runner):
        self.epoch = runner.epoch
        if (self.epoch + 1) % self.save_interval == 0:
            self.dump_json()

    def before_train_iter(self, runner):
        gt_bbox_datas = runner.data_batch['gt_bboxes'].data[0]
        self.batch_size = len(gt_bbox_datas)
        for gt_bboxes in gt_bbox_datas:
            if len(gt_bboxes.shape) < 2:
                self.shapeless2 += 1
            else:
                if gt_bboxes.shape[0] == 0:
                    self.noimg += 1
                else:
                    for gt_bbox in gt_bboxes:
                        w, h = int(gt_bbox[2] - gt_bbox[0]), int(gt_bbox[3] -
                                                                 gt_bbox[1])
                        tag = f'{w},{h}'
                        if self.size_container.get(tag, None) is None:
                            self.size_container[tag] = 1
                        else:
                            self.size_container[tag] += 1
                        self.total_sample_num += 1

    def dump_json(self):
        with open(self.out_file, 'w') as f:
            json.dump(
                {
                    'datetime:': str(datetime.now()),
                    'Batch_size': self.batch_size,
                    'Total_sample': self.total_sample_num,
                    'Noimg': self.noimg,
                    'Shapeless2': self.shapeless2,
                    'data': self.size_container
                }, f)

    # def after_run(self, runner):
    #     return super().after_run(runner)
