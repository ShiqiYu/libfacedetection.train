# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict

import torch
from mmcv.runner.hooks import HOOKS
from mmcv.runner.hooks.logger import TextLoggerHook


@HOOKS.register_module()
class YuNetTextLoggerHook(TextLoggerHook):
    """Log input size."""

    def log(self, runner):
        if 'eval_iter_num' in runner.log_buffer.output:
            # this doesn't modify runner.iter and is regardless of by_epoch
            cur_iter = runner.log_buffer.output.pop('eval_iter_num')
        else:
            cur_iter = self.get_iter(runner, inner_iter=True)

        log_dict = OrderedDict(
            mode=self.get_mode(runner),
            epoch=self.get_epoch(runner),
            iter=cur_iter)

        # only record lr of the first param group
        cur_lr = runner.current_lr()
        if isinstance(cur_lr, list):
            log_dict['lr'] = cur_lr[0]
        else:
            assert isinstance(cur_lr, dict)
            log_dict['lr'] = {}
            for k, lr_ in cur_lr.items():
                assert isinstance(lr_, list)
                log_dict['lr'].update({k: lr_[0]})

        if 'time' in runner.log_buffer.output:
            # statistic memory
            if torch.cuda.is_available():
                log_dict['memory'] = self._get_max_memory(runner)

        log_dict = dict(log_dict, **runner.log_buffer.output)

        log_dict['image_scale'] = runner.data_batch['img_metas'].data[0][0][
            'img_shape'][:2]

        self._log_info(log_dict, runner)
        self._dump_log(log_dict, runner)
        return log_dict
