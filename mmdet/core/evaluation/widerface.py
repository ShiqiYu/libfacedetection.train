"""
WiderFace evaluation code
author: wondervictor
mail: tianhengcheng@gmail.com
copyright@wondervictor
"""

from __future__ import absolute_import
import datetime
import os
import pickle

import numpy as np
import tqdm
from scipy.io import loadmat


def bbox_overlaps(boxes, query_boxes):
    n_ = boxes.shape[0]
    k_ = query_boxes.shape[0]
    overlaps = np.zeros((n_, k_), dtype=np.float)
    for k in range(k_):
        query_box_area = (query_boxes[k, 2] - query_boxes[k, 0] + 1) * (
            query_boxes[k, 3] - query_boxes[k, 1] + 1)
        for n in range(n_):
            iw = min(boxes[n, 2], query_boxes[k, 2]) - max(
                boxes[n, 0], query_boxes[k, 0]) + 1
            if iw > 0:
                ih = min(boxes[n, 3], query_boxes[k, 3]) - max(
                    boxes[n, 1], query_boxes[k, 1]) + 1
                if ih > 0:
                    box_area = (boxes[n, 2] - boxes[n, 0] + 1) * (
                        boxes[n, 3] - boxes[n, 1] + 1)
                    all_area = float(box_area + query_box_area - iw * ih)
                    overlaps[n, k] = iw * ih / all_area
    return overlaps


def bbox_overlap(a, b):
    x1 = np.maximum(a[:, 0], b[0])
    y1 = np.maximum(a[:, 1], b[1])
    x2 = np.minimum(a[:, 2], b[2])
    y2 = np.minimum(a[:, 3], b[3])
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    inter = w * h
    aarea = (a[:, 2] - a[:, 0] + 1) * (a[:, 3] - a[:, 1] + 1)
    barea = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
    o = inter / (aarea + barea - inter)
    o[w <= 0] = 0
    o[h <= 0] = 0
    return o


def np_around(array, num_decimals=0):
    return np.around(array, decimals=num_decimals)


def np_round(val, decimals=4):
    return val


def get_gt_boxes(gt_dir):
    """gt dir: (wider_face_val.mat, wider_easy_val.mat, wider_medium_val.mat,
    wider_hard_val.mat)"""

    gt_mat = loadmat(os.path.join(gt_dir, 'wider_face_val.mat'))
    hard_mat = loadmat(os.path.join(gt_dir, 'wider_hard_val.mat'))
    medium_mat = loadmat(os.path.join(gt_dir, 'wider_medium_val.mat'))
    easy_mat = loadmat(os.path.join(gt_dir, 'wider_easy_val.mat'))

    facebox_list = gt_mat['face_bbx_list']
    event_list = gt_mat['event_list']
    file_list = gt_mat['file_list']

    hard_gt_list = hard_mat['gt_list']
    medium_gt_list = medium_mat['gt_list']
    easy_gt_list = easy_mat['gt_list']

    return facebox_list, event_list, file_list, \
        hard_gt_list, medium_gt_list, easy_gt_list


def get_gt_boxes_from_txt(gt_path, cache_dir):

    cache_file = os.path.join(cache_dir, 'gt_cache.pkl')
    if os.path.exists(cache_file):
        f = open(cache_file, 'rb')
        boxes = pickle.load(f)
        f.close()
        return boxes

    f = open(gt_path, 'r')
    state = 0
    lines = f.readlines()
    lines = list(map(lambda x: x.rstrip('\r\n'), lines))
    boxes = {}
    f.close()
    current_boxes = []
    current_name = None
    for line in lines:
        if state == 0 and '--' in line:
            state = 1
            current_name = line
            continue
        if state == 1:
            state = 2
            continue

        if state == 2 and '--' in line:
            state = 1
            boxes[current_name] = np.array(current_boxes).astype('float32')
            current_name = line
            current_boxes = []
            continue

        if state == 2:
            box = [float(x) for x in line.split(' ')[:4]]
            current_boxes.append(box)
            continue

    f = open(cache_file, 'wb')
    pickle.dump(boxes, f)
    f.close()
    return boxes


def read_pred_file(filepath):

    with open(filepath, 'r') as f:
        lines = f.readlines()
        img_file = lines[0].rstrip('\n\r')
        lines = lines[2:]

    boxes = np.array(
        list(
            map(lambda x: [float(a) for a in x.rstrip('\r\n').split(' ')],
                lines))).astype('float')
    return img_file.split('/')[-1], boxes


def get_preds(pred_dir):
    events = os.listdir(pred_dir)
    boxes = dict()
    pbar = tqdm.tqdm(events)

    for event in pbar:
        pbar.set_description('Reading Predictions ')
        event_dir = os.path.join(pred_dir, event)
        event_images = os.listdir(event_dir)
        current_event = dict()
        for imgtxt in event_images:
            imgname, _boxes = read_pred_file(os.path.join(event_dir, imgtxt))
            current_event[imgname.rstrip('.jpg')] = _boxes
        boxes[event] = current_event
    return boxes


def norm_score(pred):
    """norm score pred {key: [[x1,y1,x2,y2,s]]}"""

    max_score = -1
    min_score = 2

    for _, k in pred.items():
        for _, v in k.items():
            if len(v) == 0:
                continue
            _min = np.min(v[:, -1])
            _max = np.max(v[:, -1])
            max_score = max(_max, max_score)
            min_score = min(_min, min_score)

    diff = max_score - min_score
    for _, k in pred.items():
        for _, v in k.items():
            if len(v) == 0:
                continue
            v[:, -1] = (v[:, -1] - min_score).astype(np.float64) / diff
    return pred


def image_eval(pred, gt, ignore, iou_thresh, mpp):
    """ single image evaluation
    pred: Nx5
    gt: Nx4
    ignore:
    """

    _pred = pred.copy()
    _gt = gt.copy()
    pred_recall = np.zeros(_pred.shape[0])
    recall_list = np.zeros(_gt.shape[0])
    proposal_list = np.ones(_pred.shape[0])

    _pred[:, 2] = _pred[:, 2] + _pred[:, 0]
    _pred[:, 3] = _pred[:, 3] + _pred[:, 1]
    _gt[:, 2] = _gt[:, 2] + _gt[:, 0]
    _gt[:, 3] = _gt[:, 3] + _gt[:, 1]

    gt_overlap_list = mpp.starmap(
        bbox_overlap,
        zip([_gt] * _pred.shape[0], [_pred[h] for h in range(_pred.shape[0])]))

    for h in range(_pred.shape[0]):

        gt_overlap = gt_overlap_list[h]
        max_overlap, max_idx = gt_overlap.max(), gt_overlap.argmax()

        if max_overlap >= iou_thresh:
            if ignore[max_idx] == 0:
                recall_list[max_idx] = -1
                proposal_list[h] = -1
            elif recall_list[max_idx] == 0:
                recall_list[max_idx] = 1

        r_keep_index = np.where(recall_list == 1)[0]
        pred_recall[h] = len(r_keep_index)

    return pred_recall, proposal_list


def img_pr_info(thresh_num, pred_info, proposal_list, pred_recall):
    pr_info = np.zeros((thresh_num, 2)).astype('float')
    fp = np.zeros((pred_info.shape[0], ), dtype=np.int)
    # last_info = [-1, -1]
    for t in range(thresh_num):

        thresh = 1 - (t + 1) / thresh_num
        r_index = np.where(pred_info[:, 4] >= thresh)[0]
        if len(r_index) == 0:
            pr_info[t, 0] = 0
            pr_info[t, 1] = 0
        else:
            r_index = r_index[-1]
            p_index = np.where(proposal_list[:r_index + 1] == 1)[0]
            pr_info[t, 0] = len(p_index)  # valid pred number
            pr_info[t, 1] = pred_recall[r_index]  # valid gt number

            if t > 0 and pr_info[t, 0] > pr_info[t - 1, 0] and pr_info[
                    t, 1] == pr_info[t - 1, 1]:
                fp[r_index] = 1
    return pr_info, fp


def dataset_pr_info(thresh_num, pr_curve, count_face):
    _pr_curve = np.zeros((thresh_num, 2))
    for i in range(thresh_num):
        _pr_curve[i, 0] = pr_curve[i, 1] / pr_curve[i, 0]
        _pr_curve[i, 1] = pr_curve[i, 1] / count_face
    return _pr_curve


def voc_ap(rec, prec):

    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np_round(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))
    return ap


def wider_evaluation(pred, gt_path, iou_thresh=0.5):
    # pred = get_preds(pred)
    pred = norm_score(pred)
    thresh_num = 1000
    # thresh_num = 2000
    facebox_list, event_list, file_list, hard_gt_list, \
        medium_gt_list, easy_gt_list = get_gt_boxes(gt_path)
    event_num = len(event_list)
    settings = ['easy', 'medium', 'hard']
    setting_gts = [easy_gt_list, medium_gt_list, hard_gt_list]
    from multiprocessing import Pool

    # from multiprocessing.pool import ThreadPool
    mpp = Pool(8)
    aps = [-1.0, -1.0, -1.0]
    print('')
    for setting_id in range(3):
        ta = datetime.datetime.now()
        iou_th = iou_thresh
        gt_list = setting_gts[setting_id]
        count_face = 0
        pr_curve = np.zeros((thresh_num, 2)).astype('float')
        # [hard, medium, easy]
        # high_score_count = 0
        # high_score_fp_count = 0
        for i in range(event_num):
            event_name = str(event_list[i][0][0])
            img_list = file_list[i][0]
            pred_list = pred[event_name]
            sub_gt_list = gt_list[i][0]
            # img_pr_info_list = np.zeros((len(img_list), thresh_num, 2))
            gt_bbx_list = facebox_list[i][0]

            for j in range(len(img_list)):
                img_name = str(img_list[j][0][0])
                pred_info = pred_list[img_name]

                gt_boxes = gt_bbx_list[j][0].astype('float')
                keep_index = sub_gt_list[j][0]
                count_face += len(keep_index)

                if len(gt_boxes) == 0 or len(pred_info) == 0:
                    continue

                ignore = np.zeros(gt_boxes.shape[0], dtype=np.int)
                if len(keep_index) != 0:
                    ignore[keep_index - 1] = 1
                pred_info = np_round(pred_info, 1)

                gt_boxes = np_round(gt_boxes)
                pred_recall, proposal_list = image_eval(
                    pred_info, gt_boxes, ignore, iou_th, mpp)

                _img_pr_info, fp = img_pr_info(thresh_num, pred_info,
                                               proposal_list, pred_recall)

                pr_curve += _img_pr_info
        pr_curve = dataset_pr_info(thresh_num, pr_curve, count_face)
        propose = pr_curve[:, 0]
        recall = pr_curve[:, 1]
        for srecall in np.arange(0.1, 1.0001, 0.1):
            rindex = len(np.where(recall <= srecall)[0]) - 1
            rthresh = 1.0 - float(rindex) / thresh_num
            print('Recall-Precision-Thresh:', recall[rindex], propose[rindex],
                  rthresh)

        ap = voc_ap(recall, propose)
        aps[setting_id] = ap
        tb = datetime.datetime.now()
        print('%s cost %.4f seconds, ap: %.5f' %
              (settings[setting_id], (tb - ta).total_seconds(), ap))

    return aps


def get_widerface_gts(gt_path):
    facebox_list, event_list, file_list, hard_gt_list, \
        medium_gt_list, easy_gt_list = get_gt_boxes(gt_path)
    event_num = len(event_list)

    # settings = ['easy', 'medium', 'hard']
    setting_gts = [easy_gt_list, medium_gt_list, hard_gt_list]
    all_results = []
    for setting_id in range(3):
        results = {}
        gt_list = setting_gts[setting_id]
        count_face = 0
        for i in range(event_num):
            event_name = str(event_list[i][0][0])
            img_list = file_list[i][0]
            sub_gt_list = gt_list[i][0]
            # img_pr_info_list = np.zeros((len(img_list), thresh_num, 2))
            gt_bbx_list = facebox_list[i][0]
            results[event_name] = {}

            for j in range(len(img_list)):

                gt_boxes = gt_bbx_list[j][0].astype('float').copy()
                gt_boxes[:, 2] += gt_boxes[:, 0]
                gt_boxes[:, 3] += gt_boxes[:, 1]
                keep_index = sub_gt_list[j][0].copy()
                count_face += len(keep_index)

                if len(gt_boxes) == 0:
                    results[event_name][str(img_list[j][0][0])] = np.empty(
                        (0, 4))
                    continue
                keep_index -= 1
                keep_index = keep_index.flatten()

                gt_boxes = np_round(gt_boxes)[keep_index, :]

                results[event_name][str(img_list[j][0][0])] = gt_boxes
        all_results.append(results)
    return all_results


# if __name__ == '__main__':

#     parser = argparse.ArgumentParser()
#     parser.add_argument('-p', '--pred', default='')
#     parser.add_argument('-g', '--gt', default='./ground_truth/')

#     args = parser.parse_args()
#     evaluation(args.pred, args.gt)
