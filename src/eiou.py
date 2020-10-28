import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

__all__ = ['eiou_loss']

BBOX_XFORM_CLIP = np.log(1000. / 16.)

def _decode(deltas, variance=[0.1, 0.2]):
    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]
    dw = torch.clamp(dw, max=BBOX_XFORM_CLIP)
    dh = torch.clamp(dh, max=BBOX_XFORM_CLIP)

    pctrx = dx * variance[0]
    pctry = dy * variance[0]
    pw = torch.exp(dw * variance[1])
    ph = torch.exp(dh * variance[1])

    x1 = pctrx - 0.5*pw
    y1 = pctry - 0.5*ph
    x2 = pctrx + 0.5*pw
    y2 = pctry + 0.5*ph

    return x1.view(-1), y1.view(-1), x2.view(-1), y2.view(-1)

def eiou_loss(input, target, variance=[0.1, 0.2], smooth_point=0.2, reduction='sum'):
    '''EIoU Implementation
    '''
    px1, py1, px2, py2 = _decode(input, variance)
    tx1, ty1, tx2, ty2 = _decode(target, variance)

    # extent top left
    ex1 = torch.min(px1, tx1)
    ey1 = torch.min(py1, ty1)
    # intersection coordinates
    ix1 = torch.max(px1, tx1)
    iy1 = torch.max(py1, ty1)
    ix2 = torch.min(px2, tx2)
    iy2 = torch.min(py2, ty2)
    # extra
    xmin = torch.min(ix1, ix2)
    ymin = torch.min(iy1, iy2)
    # xmax = ix2
    # ymax = iy2

    # Intersection
    intersection = (ix2 - ex1) * (iy2 - ey1) +   \
                   (xmin - ex1) * (ymin - ey1) - \
                   (ix1 - ex1) * (iy2 - ey1) -   \
                   (ix2 - ex1) * (iy1 - ey1)
    # Union
    union = (px2 - px1) * (py2 - py1) + \
            (tx2 - tx1) * (ty2 - ty1) - intersection + 1e-7
    # IoU
    iou = intersection / union

    # EIoU
    smooth_sign = (iou < smooth_point).detach().float()
    eiou = 0.5 * smooth_sign * ((1 - iou) ** 2) / smooth_point + \
           (1 - smooth_point) * ((1 - iou) - 0.5 * smooth_point)
    eiou *= union.detach()
    if reduction is None:
        l = eiou
    elif reduction.lower() == 'sum':
        l = eiou.sum(0)
    elif reduction.lower() == 'mean':
        l = eiou.sum(0) / input.size(0)
    else:
        raise NotImplementedError()
    return l