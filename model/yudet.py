import torch
import torch.nn as nn
import cv2
from .nets.layers import PriorBox
from .nets.yunet import Yunet
from .nets.yuhead import build_head
from .losses.multiboxloss import MultiBoxLoss
from .src.utils import decode

class YuDetectNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_classes = cfg['model']['head']['num_classes']
        self.num_landmarks = cfg['model']['head']['num_landmarks']
        self.out_factor = (4 + self.num_landmarks * 2 + self.num_classes + 1)
        self.num_ratio = len(cfg['model']['anchor']['ratio'])
        self.activation_type = cfg['model'].get('activation_type', 'relu')
        self.backbone = Yunet(
            cfg_layers=cfg['model']['backbone']['layers'],
            activation_type=self.activation_type)

        self.head = build_head(
            name=cfg['model']['head'].get('type', None),
            in_channels=cfg['model']['head']['in_channels'],
            out_channels=[len(x) * self.out_factor * self.num_ratio for x in cfg['model']['anchor']['min_sizes']],
            activation_type=self.activation_type 
        ) 

        self.anchor_generator = PriorBox(
            min_sizes=cfg['model']['anchor']['min_sizes'],
            steps=cfg['model']['anchor']['steps'],
            clip=cfg['model']['anchor']['clip'],
            ratio=cfg['model']['anchor']['ratio']
        )
        self.criterion = MultiBoxLoss(
            num_classes=self.num_classes,
            iou_threshold=cfg['model']['loss']['overlap_thresh'],
            negpos_ratio=cfg['model']['loss']['neg_pos'],
            variance=cfg['model']['loss']['variance'],
            smooth_point=cfg['model']['loss']['smooth_point']
        )
        self.anchors_set = {}
        self.cfg = cfg
    
    def forward(self, x):
        self.img_size = x.shape[-2:]
        feats = self.backbone(x)
        outs = self.head(feats)
        head_data=[(x.permute(0, 2, 3, 1).contiguous()) for x in outs]
        head_data = torch.cat([o.view(o.size(0), -1) for o in head_data], dim=1)
        head_data = head_data.view(head_data.size(0), -1, self.out_factor)

        loc_data = head_data[:, :, 0 : 4 + self.num_landmarks * 2]
        conf_data = head_data[:, :, -self.num_classes - 1 : -1]
        iou_data = head_data[:,:, -1:]
        output = (loc_data, conf_data, iou_data)
        return output

    def get_anchor(self, img_shape=None):
        if img_shape is None:
            img_shape = self.img_size
        if len(img_shape) == 3:
            img_shape = img_shape[:2]
        if self.anchors_set.__contains__(img_shape):
            return self.anchors_set[img_shape]
        else:
            anchors = self.anchor_generator(img_shape)
            self.anchors_set[img_shape] = anchors
            return anchors

    def loss(self, predictions, targets):
        priors = self.get_anchor().cuda()
        loss_bbox_eiou, loss_iouhead_smoothl1, loss_lm_smoothl1, loss_cls_ce = \
            self.criterion(predictions, priors, targets)
        loss_bbox_eiou *= self.cfg['model']['loss']['weight_bbox']
        loss_iouhead_smoothl1 *= self.cfg['model']['loss']['weight_iouhead']
        loss_lm_smoothl1 *=  self.cfg['model']['loss']['weight_lmds']
        loss_cls_ce *= self.cfg['model']['loss']['weight_cls']
        return (loss_bbox_eiou, loss_iouhead_smoothl1, loss_lm_smoothl1, loss_cls_ce)

    def inference(self, img, scale, without_landmarks=True, device='cuda:0'):
        if scale != 1.:
            img = cv2.resize(img, None, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        priors = self.get_anchor(img.shape).to(device)
        h, w, _ = img.shape    
        img = torch.from_numpy(img).to(device)
        img = img.permute(2, 0, 1).unsqueeze(0)
        img = img.float()
        loc, conf, iou = self(img)
        conf = torch.softmax(conf.squeeze(0), dim=-1)

        boxes = decode(loc.squeeze(0), priors.data, self.cfg['model']['loss']['variance'])
        box_dim = 4 if without_landmarks else (4 + self.num_landmarks * 2)
        boxes = boxes[:, :box_dim]
        boxes[:, 0::2] = boxes[:, 0::2] * w / scale
        boxes[:, 1::2] = boxes[:, 1::2] * h / scale
        cls_scores = conf.squeeze(0)[:, 1]
        iou_scores = iou.squeeze(0)[:, 0]

        iou_scores = torch.clamp(iou_scores, min=0., max=1.)
        scores = torch.sqrt(cls_scores * iou_scores)
        score_mask = scores > self.cfg['test']['confidence_threshold']
        boxes = boxes[score_mask]
        scores = scores[score_mask]
        _boxes = boxes[:, :4].clone()
        _boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        _boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        iou_thresh = self.cfg['test']['nms_threshold']
        keep_idx = cv2.dnn.NMSBoxes(
                    bboxes=_boxes.tolist(), 
                    scores=scores.tolist(), 
                    score_threshold=self.cfg['test']['confidence_threshold'], 
                    nms_threshold=iou_thresh, eta=1, 
                    top_k=self.cfg['test']['top_k']
        )
        if len(keep_idx) > 0:
            keep_idx = keep_idx.reshape(-1)
            boxes = boxes[keep_idx]
            scores = scores[keep_idx]
            dets = torch.cat([boxes, scores[:, None]], dim=-1)
            dets = dets[:self.cfg['test']['keep_top_k']]
        else:
            dets = torch.empty((0, box_dim + 1)).to(device)
        return dets

    def export_cpp(self, filename):
        '''This function can export CPP data file for libfacedetection'''
        result_str = '// Auto generated data file\n'
        result_str += '// Copyright (c) 2018-2021, Shiqi Yu, all rights reserved.\n'
        result_str += '#include "facedetectcnn.h" \n\n'
        
        DPConvs = []
        DPConvs += [self.backbone.model0]
        DPConvs += [self.backbone.model1.conv1, self.backbone.model1.conv2]
        DPConvs += [self.backbone.model2.conv1, self.backbone.model2.conv2]
        DPConvs += [self.backbone.model3.conv1, self.backbone.model3.conv2]
        DPConvs += [self.backbone.model4.conv1, self.backbone.model4.conv2]
        DPConvs += [self.backbone.model5.conv1, self.backbone.model5.conv2]
        DPConvs += [self.backbone.model6.conv1, self.backbone.model6.conv2]
        # for (l, c) in zip(self.loc, self.conf):
        #     DPConvs += [l.conv1, l.conv2]
        #     DPConvs += [c.conv1, c.conv2]
        for layers in self.head.head:
            DPConvs += [layers.conv1, layers.conv2]

        # convert to a string
        num_conv = len(DPConvs)
        # the first conv_head layer
        # result_str += convert_param2string(combine_conv_bn(DPConvs[0], self.model0.bn1), 'f0', False, True)
        # result_str += DPConvs[0].convert_to_cppstring()
        # the rest depthwise+pointwise conv layers
        result_str += DPConvs[0].convert_to_cppstring('f')
        for idx in range(1, num_conv):
            rs = DPConvs[idx].convert_to_cppstring('f' + str(idx + 1))
            result_str += rs
            result_str += '\n'

        result_str += 'ConvInfoStruct param_pConvInfo[' + str(num_conv*2 + 1) + '] = { \n'

        result_str += '   {32, ' + str(DPConvs[0].conv1.out_channels) +', false, true, true, f0_weight, f0_bias},\n'
        result_str += '   {'+ str(DPConvs[0].conv2.in_channels) + ', ' + str(DPConvs[0].conv2.out_channels) +', false, true, false, f1_1_weight, f1_1_bias},\n'
        result_str += '   {'+ str(DPConvs[0].conv2.out_channels) + ', ' + str(DPConvs[0].conv2.out_channels) +', true, false, true, f1_2_weight, f1_2_bias},\n'
        
        for idx in range(1, num_conv):
            result_str += ('    {' +
                           str(DPConvs[idx].in_channels) + ', ' +
                           str(DPConvs[idx].out_channels) + ', ' +
                           'false, ' + # is_depthwise 
                           'true, ' + # is_pointwise
                           'false, ' + # with_relu
                           'f' + str(idx + 1) + '_1_weight' + ', ' +
                           'f' + str(idx + 1) + '_1_bias' +
                           '}')

            result_str += ','
            result_str += '\n'

            with_relu = 'false, '
            if(DPConvs[idx].withBNRelu):
                with_relu = 'true, '

            result_str += ('    {' +
                           str(DPConvs[idx].out_channels) + ', ' +
                           str(DPConvs[idx].out_channels) + ', ' +
                           'true, ' + # is_depthwise 
                           'false, ' + # is_pointwise
                           with_relu + # with_relu
                           'f' + str(idx + 1) + '_2_weight' + ', ' +
                           'f' + str(idx + 1) + '_2_bias' +
                           '}')

            if (idx < num_conv - 1):
                result_str += ','
            result_str += '\n'

        result_str += '};\n'
        

        # write the content to a file
        #print(result_str)
        with open(filename, 'w') as f:
            f.write(result_str)
            f.close()

        return 0 