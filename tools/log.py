import os
import logging
import time
class Logger(object):
    def __init__(self, cfg, mode) -> None:
        super().__init__()
        assert mode in ['train', 'test']

        self.logger = logging.getLogger(cfg.get('name', 'LOG'))
        self.logger.setLevel(logging.DEBUG)

        if mode == 'train':
            self.cfg = cfg['train']
            self.is_log = self.cfg['logger']['logfile']
            self.is_sysout = self.cfg['logger']['sysout']
            self.is_tb = self.cfg['logger']['tensorboard']

            if self.is_log:
                file_hander = logging.FileHandler(os.path.join(self.cfg['log_dir'], f'{mode}.log'), mode='a')
                file_fmt = logging.Formatter('[%(asctime)s] - %(message)s', datefmt='%Y%m%d %H:%M:%S')
                file_hander.setFormatter(file_fmt)
                file_hander.setLevel(logging.DEBUG)
                self.logger.addHandler(file_hander)

            if self.is_sysout:
                sys_hander = logging.StreamHandler()
                sys_hander.setLevel(logging.DEBUG)
                sys_fmt = logging.Formatter('[%(asctime)s] - %(message)s', datefmt='(%d) %H:%M:%S')
                sys_hander.setFormatter(sys_fmt)
                self.logger.addHandler(sys_hander)

            if self.is_tb:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_logger = SummaryWriter(os.path.join(self.cfg['log_dir'], 'tb_log'))    

            self.cur_epoch = self.cfg['start_epoch']
            self.max_epoch = self.cfg['max_epochs']
            self.flag_meter_initial = False
            self.meters = None
            self.epoch = 0
            self.log_interval = self.cfg['logger']['log_interval']

        elif mode == 'test':
            self.cfg = cfg['test']
            self.is_log = self.cfg['logger']['logfile']
            self.is_sysout = self.cfg['logger']['sysout']
            if self.is_log:
                file_hander = logging.FileHandler(os.path.join(self.cfg['log_dir'], f'{mode}.log'), mode='a')
                file_fmt = logging.Formatter('[%(asctime)s] - %(message)s', datefmt='%Y%m%d %H:%M:%S')
                file_hander.setFormatter(file_fmt)
                file_hander.setLevel(logging.DEBUG)
                self.logger.addHandler(file_hander)

            if self.is_sysout:
                sys_hander = logging.StreamHandler()
                sys_hander.setLevel(logging.DEBUG)
                sys_fmt = logging.Formatter('[%(asctime)s] - %(message)s', datefmt='(%d) %H:%M:%S')
                sys_hander.setFormatter(sys_fmt)
                self.logger.addHandler(sys_hander)


    def info(self, msg):
        self.logger.info(msg)
    
    # override stdout
    def write(self, msg):
        self.logger.info(msg)    
    def flush(self):
        pass

    def update(self, msg_dict:dict):
        if not self.flag_meter_initial:
            self.meters = self.meter_initial(msg_dict)
            self.flag_meter_initial = True

        it = msg_dict['iter']
        max_it = msg_dict['max_iter']
        lr = msg_dict['lr']

        for k in self.meters.keys():
            self.meters[k].update(msg_dict[k])
            if self.is_tb:
                self.tb_logger.add_scalar(
                    tag=f'Iter/{k}',
                    scalar_value=msg_dict[k],
                    global_step=it + max_it * self.epoch
                )

        if it % self.log_interval == 0 and it != 0:
            self.logger.info(f'[{self.epoch} : {it} / {max_it}] [LR: {lr:.6f}]\t' + 
                            "\t".join([f'{k}: {v.val:.3f} ({v.avg:.3f})' for k, v in self.meters.items()]) + 
                            "\t" + "\t".join([f'{k}: {v}' for k, v in msg_dict['tag'].items()]))
        
        if it == 0:
            if self.is_tb:
                for k, v in self.meters.items():
                    self.tb_logger.add_scalar(
                        tag=f'Epoch/{k}',
                        scalar_value=v.avg,
                        global_step=self.epoch
                    )
            self.epoch += 1

    def meter_initial(self, msg):
        meters = {}
        for k in msg.keys():
            if k.startswith('loss'):
                meters[k] = dataMeter()
        return meters

class dataMeter(object):
    def __init__(self) -> None:
        super().__init__()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.num = 0
    
    def update(self, value):
        self.val = value
        self.sum += value
        self.num += 1
        self.avg = self.sum / self.num

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.num = 0