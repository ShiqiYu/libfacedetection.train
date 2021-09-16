#-*- coding:utf-8 -*-
import os
import os.path
import sys
import cv2
import random
import torch
import torch.utils.data as data
import numpy as np
from utils import matrix_iof

#import for dali dataloader
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
from nvidia.dali import types
from nvidia.dali import math as math
from nvidia.dali.plugin.base_iterator import _DaliBaseIterator
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
import ctypes

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

WIDER_CLASSES = ('__background__', 'face')


def _crop(image, boxes, labels, img_dim):
    height, width, _ = image.shape
    pad_image_flag = True

    for _ in range(250):
        if random.uniform(0, 1) <= 0.2:
            scale = 1
        else:
            scale = random.uniform(0.3, 1.)
        short_side = min(width, height)
        w = int(scale * short_side)
        h = w

        if width == w:
            l = 0
        else:
            l = random.randrange(width - w)
        if height == h:
            t = 0
        else:
            t = random.randrange(height - h)
        roi = np.array((l, t, l + w, t + h))

        value = matrix_iof(boxes, roi[np.newaxis])
        flag = (value >= 1)
        if not flag.any():
            continue

        centers = (boxes[:, 0:2] + boxes[:, 2:4]) / 2
        mask_a = np.logical_and(roi[:2] < centers, centers < roi[2:]).all(axis=1)
        boxes_t = boxes[mask_a].copy()
        labels_t = labels[mask_a].copy()

        if boxes_t.shape[0] == 0:
            continue

        #the cropped image
        image_t = image[roi[1]:roi[3], roi[0]:roi[2]]
        #to avoid the TL corner being out of the roi boundary
        boxes_t[:, 0:2] = np.maximum(boxes_t[:, :2], roi[:2])
        #to avoid the BR corner being out of the roi boundary
        boxes_t[:, 2:4] = np.minimum(boxes_t[:, 2:4], roi[2:4])
        #shift all points (x,y) according to the TL of the roi
        boxes_t[:, 0::2] -= roi[0]
        boxes_t[:, 1::2] -= roi[1]

        # make sure that the cropped image contains at least one face > 8 pixel at training image scale
        b_w_t = (boxes_t[:, 2] - boxes_t[:, 0] + 1) / w * img_dim
        b_h_t = (boxes_t[:, 3] - boxes_t[:, 1] + 1) / h * img_dim
        mask_b = np.minimum(b_w_t, b_h_t) > 8.0
        boxes_t = boxes_t[mask_b]
        labels_t = labels_t[mask_b]

        if boxes_t.shape[0] == 0:
            continue

        pad_image_flag = False

        return image_t, boxes_t, labels_t, pad_image_flag
    return image, boxes, labels, pad_image_flag


def _distort(image):

    def _convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    image = image.copy()

    if random.randrange(2):

        #brightness distortion
        if random.randrange(2):
            _convert(image, beta=random.uniform(-32, 32))

        #contrast distortion
        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.5, 1.5))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        #saturation distortion
        if random.randrange(2):
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

        #hue distortion
        if random.randrange(2):
            tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    else:

        #brightness distortion
        if random.randrange(2):
            _convert(image, beta=random.uniform(-32, 32))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        #saturation distortion
        if random.randrange(2):
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

        #hue distortion
        if random.randrange(2):
            tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        #contrast distortion
        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.5, 1.5))

    return image


def _expand(image, boxes, fill, p):
    if random.randrange(2):
        return image, boxes

    height, width, depth = image.shape

    scale = random.uniform(1, p)
    w = int(scale * width)
    h = int(scale * height)

    left = random.randint(0, w - width)
    top = random.randint(0, h - height)

    boxes_t = boxes.copy()
    boxes_t[:, :2] += (left, top)
    boxes_t[:, 2:] += (left, top)
    expand_image = np.empty(
        (h, w, depth),
        dtype=image.dtype)
    expand_image[:, :] = fill
    expand_image[top:top + height, left:left + width] = image
    image = expand_image

    return image, boxes_t


def _mirror(image, boxes):
    _, width, _ = image.shape
    if random.randrange(2):
        image = image[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]
    return image, boxes


def _pad_to_square(image, rgb_mean, pad_image_flag):
    if not pad_image_flag:
        return image
    height, width, _ = image.shape
    long_side = max(width, height)
    image_t = np.empty((long_side, long_side, 3), dtype=image.dtype)
    image_t[:, :] = rgb_mean
    image_t[0:0 + height, 0:0 + width] = image
    return image_t


def _resize_subtract_mean(image, insize, rgb_mean):
    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    interp_method = interp_methods[random.randrange(5)]
    image = cv2.resize(image, (insize, insize), interpolation=interp_method)
    image = image.astype(np.float32)
    image -= rgb_mean
    return image.transpose(2, 0, 1)


class PreProc(object):

    def __init__(self, img_dim, rgb_means):
        self.img_dim = img_dim
        self.rgb_means = rgb_means

    def __call__(self, image, targets):
        assert targets.shape[0] > 0, "this image does not have gt"

        boxes = targets[:, :-1].copy()
        labels = targets[:, -1].copy()

        image_t, boxes_t, labels_t, pad_image_flag = _crop(image, boxes, labels, self.img_dim)
        image_t = _distort(image_t)
        image_t = _pad_to_square(image_t,self.rgb_means, pad_image_flag)

        ##since the landmarks should also be flipped (left eye<->right eye)
        ##it's too complex. We disable _mirror_ operation here
        #image_t, boxes_t = _mirror(image_t, boxes_t)

        #convert (x,y) to range [0,1]
        height, width, _ = image_t.shape
        boxes_t[:, 0::2] /= width
        boxes_t[:, 1::2] /= height

        image_t = _resize_subtract_mean(image_t, self.img_dim, self.rgb_means)

        labels_t = np.expand_dims(labels_t, 1)
        targets_t = np.hstack((boxes_t, labels_t))

        return image_t, targets_t


class AnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=True):
        self.class_to_ind = class_to_ind or dict(
            zip(WIDER_CLASSES, range(len(WIDER_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = np.empty((0, 15))
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            # has_lm = int(obj.find('has_lm').text)

            # get face rect
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text)
                bndbox.append(cur_pt)

            # get face landmark
            if int(obj.find('has_lm').text.strip()) == 1:
                lm = obj.find('lm')
                pts = ['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'x5', 'y5']
                for i, pt in enumerate(pts):
                    xy_value = float(lm.find(pt).text)
                    bndbox.append(xy_value)
            else:  # append 10 zeros
                for i in range(10):
                    bndbox.append(0)

            # label 0 or 1 (bk or face)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)

            res = np.vstack(
                (res, bndbox))  # [xmin, ymin, xmax, ymax, label_ind, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5]
        return res

class RetinaFaceDataset(data.Dataset):
    '''The training dataset is re-labeled by RetinaFace.
    '''

    def __init__(self, root, img_dim, rgb_mean, score=0.5, iou=0.1):
        self.root = root
        self.preproc = PreProc(img_dim, rgb_mean)

        self.anno_path = os.path.join(self.root, 'train_label', '{}.txt')
        self.img_path = os.path.join(self.root, 'WIDER_train', 'images', '{}.jpg')

        self.ids = list()
        with open(os.path.join(self.root, 'train_label', 'anno_list.txt'), 'r') as f:
            for line in f:
                line = line.strip()
                if '.txt' in line:
                    line = line.replace('.txt', '')
                self.ids.append(line)

    def load_anno(self, anno_filepath, scale_thresh=10):
        '''In each anno_filepath:
        img_subpath
        face_num
        face1 (x1, y1, w, h, lm0_x, lm0_y, lm1_x, lm1_y, lm2_x, lm2_y, lm3_x, lm3_y, lm4_x, lm4_y, score)
        face2 (x1, y1, w, h, lm0_x, lm0_y, lm1_x, lm1_y, lm2_x, lm2_y, lm3_x, lm3_y, lm4_x, lm4_y, score)
        ...

        Load it into a numpy array nx(x1, y1, x2, y2, lm0_x, lm0_y, lm1_x, lm1_y, lm2_x, lm2_y, lm3_x, lm3_y, lm4_x, lm4_y)
        '''
        target = np.empty(shape=(0, 15), dtype=np.float32)
        with open(anno_filepath, 'r') as f:
            lines = f.readlines()
            face_num = int(lines[1])
            
            for idx in range(face_num):
                coords = [float(c) for c in lines[2+idx].split(' ')]
                if coords[2] * coords[3] < scale_thresh * scale_thresh:
                    continue
                coords[2] = coords[0] + coords[2] # x2 = x1 + w
                coords[3] = coords[1] + coords[3] # y2 = y1 + h
                coords[-1] = 1                    # label = 1
                target = np.vstack(
                    (target, coords)
                )

            if target.shape[0] == 0:
                for idx in range(face_num):
                    coords = [float(c) for c in lines[2+idx].split(' ')]
                    coords[2] = coords[0] + coords[2] # x2 = x1 + w
                    coords[3] = coords[1] + coords[3] # y2 = y1 + h
                    coords[-1] = 1                    # label = 1
                    target = np.vstack(
                        (target, coords)
                    )

        return target

    def __getitem__(self, idx):
        img_id = self.ids[idx]

        # load annotation
        target = self.load_anno(self.anno_path.format(img_id))

        # load image
        img = cv2.imread(self.img_path.format(img_id), cv2.IMREAD_COLOR)

        # preprocess
        if self.preproc is not None:
            img, target = self.preproc(img, target)

        return torch.from_numpy(img), target

    def __len__(self):
        return len(self.ids)

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)


class WiderfacePipeline_COCOformat(Pipeline):
    def __init__(self, batch_size, device_id, 
                file_root, annotations_file, num_gpus, input_dim = 320.,
                num_threads=1, seed=-1, random_shuffle=False, shuffle_after_epoch=False):
        super(WiderfacePipeline_COCOformat, self).__init__(batch_size=batch_size, device_id=device_id,
                                           num_threads=num_threads, seed = seed)

        if torch.distributed.is_initialized():
            shard_id = torch.distributed.get_rank()
        else:
            shard_id = 0

        self.share_id = shard_id
        self.num_gpus = num_gpus
        self.file_root = file_root
        self.annotation_file = annotations_file
        self.input_dim = float(input_dim)
        self.random_shuffle = random_shuffle
        self.shuffle_after_epoch = shuffle_after_epoch

    def define_graph(self):
        inputs, bboxes, labels, polygons, vertices = fn.readers.coco(
                                            file_root=self.file_root,
                                            annotations_file=self.annotation_file,
                                            skip_empty=True,
                                            shard_id=self.share_id,
                                            num_shards=self.num_gpus,
                                            ratio=True,
                                            ltrb=True,
                                            polygon_masks = True,
                                            random_shuffle=self.random_shuffle,
                                            shuffle_after_epoch=self.shuffle_after_epoch,
                                            name="Reader")

        input_shape = fn.slice(fn.cast(fn.peek_image_shape(inputs), dtype=types.INT32), 0, 2, axes=[0])
        h = fn.slice(input_shape, 0, 1, axes = [0], dtype=types.FLOAT)
        w = fn.slice(input_shape, 1, 1, axes = [0], dtype=types.FLOAT)
        short_side = math.min(w, h)        
        scale = fn.random.uniform(range=[0.3, 1.])
        crop_side = fn.cast(math.ceil(scale * short_side), dtype=types.INT32)    
        crop_shape = fn.cat(crop_side, crop_side)
        anchor_rel, shape_rel, bboxes, labels, bbox_indices = fn.random_bbox_crop(
                        bboxes,
                        labels,
                        input_shape=input_shape,
                        crop_shape=crop_shape,
                        shape_layout="HW",
                        thresholds=[0.],            # No minimum intersection-over-union, for demo purposes
                        allow_no_crop=False,        # No-crop is disallowed, for demo purposes 
                        seed=-1,                    # Fixed random seed for deterministic results
                        bbox_layout="xyXY",         # left, top, right, back
                        output_bbox_indices=True,   # Output indices of the filtered bounding boxes
                        total_num_attempts=1024,
        )
        polygons, vertices = fn.segmentation.select_masks(
            bbox_indices, polygons, vertices
        )
        images = fn.decoders.image_slice(
            inputs, anchor_rel, shape_rel, normalized_anchor=False, normalized_shape=False, device='mixed'
        )
        images = fn.color_space_conversion(images, image_type=types.RGB, output_type=types.BGR)
        MT_1_vertices = fn.transforms.crop(
            to_start=(0.0, 0.0), to_end=fn.cat(w, h)
        )    
        MT_2_vertices = fn.transforms.crop(
            from_start=anchor_rel, from_end=(anchor_rel + shape_rel),
            to_start=(0.0, 0.0), to_end=(1., 1.)
        )    
        vertices = fn.coord_transform(fn.coord_transform(vertices, MT=MT_1_vertices), MT=MT_2_vertices)    
        targets = fn.cat( bboxes, fn.reshape(vertices, shape=[-1, 10]), axis=1)

        interp_methods = [types.INTERP_LINEAR, types.INTERP_CUBIC, types.INTERP_LANCZOS3, types.INTERP_GAUSSIAN, types.INTERP_NN, types.INTERP_TRIANGULAR]
        interp_method = fn.random.uniform(values=[int(x) for x in interp_methods], dtype=types.INT32)
        interp_method = fn.reinterpret(interp_method, dtype=types.INTERP_TYPE)
        images = fn.resize(images, dtype=types.FLOAT, size=self.input_dim, interp_type=interp_method)

        labels = labels.gpu()
        targets = targets.gpu()
        return (images, targets, labels)

to_torch_type = {
    np.dtype(np.float32) : torch.float32,
    np.dtype(np.float64) : torch.float64,
    np.dtype(np.float16) : torch.float16,
    np.dtype(np.uint8)   : torch.uint8,
    np.dtype(np.int8)    : torch.int8,
    np.dtype(np.int16)   : torch.int16,
    np.dtype(np.int32)   : torch.int32,
    np.dtype(np.int64)   : torch.int64
}

def feed_ndarray(dali_tensor, arr):
    """
    Copy contents of DALI tensor to pyTorch's Tensor.

    Parameters
    ----------
    `dali_tensor` : nvidia.dali.backend.TensorCPU or nvidia.dali.backend.TensorGPU
                    Tensor from which to copy
    `arr` : torch.Tensor
            Destination of the copy
    """
    assert dali_tensor.shape() == list(arr.size()), \
            ("Shapes do not match: DALI tensor has size {0}"
            ", but PyTorch Tensor has size {1}".format(dali_tensor.shape(), list(arr.size())))
    #turn raw int to a c void pointer
    c_type_pointer = ctypes.c_void_p(arr.data_ptr())
    dali_tensor.copy_to_external(c_type_pointer)
    return arr

class DALIGenericIterator(_DaliBaseIterator):
    """
    General DALI iterator for PyTorch. It can return any number of
    outputs from the DALI pipeline in the form of PyTorch's Tensors.

    Please keep in mind that Tensors returned by the iterator are
    still owned by DALI. They are valid till the next iterator call.
    If the content needs to be preserved please copy it to another tensor.

    Parameters
    ----------
    pipelines : list of nvidia.dali.Pipeline
                List of pipelines to use
    output_map : list of str
                 List of strings which maps consecutive outputs
                 of DALI pipelines to user specified name.
                 Outputs will be returned from iterator as dictionary
                 of those names.
                 Each name should be distinct
    size : int, default = -1
           Number of samples in the shard for the wrapped pipeline (if there is more than one it is a sum)
           Providing -1 means that the iterator will work until StopIteration is raised
           from the inside of iter_setup(). The options `last_batch_policy`, `last_batch_padded` and
           `auto_reset` don't work in such case. It works with only one pipeline inside
           the iterator.
           Mutually exclusive with `reader_name` argument
    reader_name : str, default = None
                Name of the reader which will be queried to the shard size, number of shards and
                all other properties necessary to count properly the number of relevant and padded
                samples that iterator needs to deal with. It automatically sets `last_batch_policy` to
                PARTIAL when the FILL is used, and `last_batch_padded` accordingly to match
                the reader's configuration
    auto_reset : bool, optional, default = False
                 Whether the iterator resets itself for the next epoch
                 or it requires reset() to be called separately.
    dynamic_shape: bool, optional, default = False
                 Whether the shape of the output of the DALI pipeline can
                 change during execution. If True, the pytorch tensor will be resized accordingly
                 if the shape of DALI returned tensors changes during execution.
                 If False, the iterator will fail in case of change.
    fill_last_batch : bool, optional, default = None
                **Deprecated** Please use ``last_batch_policy`` instead

                Whether to fill the last batch with data up to 'self.batch_size'.
                The iterator would return the first integer multiple
                of self._num_gpus * self.batch_size entries which exceeds 'size'.
                Setting this flag to False will cause the iterator to return
                exactly 'size' entries.
    last_batch_policy : default = FILL
                What to do with the last batch when there is no enough samples in the epoch
                to fully fill it. See :meth:`nvidia.dali.plugin.base_iterator.LastBatchPolicy`
    last_batch_padded : bool, optional, default = False
                Whether the last batch provided by DALI is padded with the last sample
                or it just wraps up. In the conjunction with ``last_batch_policy`` it tells
                if the iterator returning last batch with data only partially filled with
                data from the current epoch is dropping padding samples or samples from
                the next epoch (it doesn't literally drop but sets ``pad`` field of ndarray
                so the following code could use it to drop the data). If set to ``False`` next
                epoch will end sooner as data from it was consumed but dropped. If set to
                True next epoch would be the same length as the first one. For this to happen,
                the option `pad_last_batch` in the reader needs to be set to True as well.
                It is overwritten when `reader_name` argument is provided
    prepare_first_batch : bool, optional, default = True
                Whether DALI should buffer the first batch right after the creation of the iterator,
                so one batch is already prepared when the iterator is prompted for the data

    Example
    -------
    With the data set ``[1,2,3,4,5,6,7]`` and the batch size 2:

    last_batch_policy = PARTIAL, last_batch_padded = True  -> last batch = ``[7]``, next iteration will return ``[1, 2]``

    last_batch_policy = PARTIAL, last_batch_padded = False -> last batch = ``[7]``, next iteration will return ``[2, 3]``

    last_batch_policy = FILL, last_batch_padded = True   -> last batch = ``[7, 7]``, next iteration will return ``[1, 2]``

    last_batch_policy = FILL, last_batch_padded = False  -> last batch = ``[7, 1]``, next iteration will return ``[2, 3]``

    last_batch_policy = DROP, last_batch_padded = True   -> last batch = ``[5, 6]``, next iteration will return ``[1, 2]``

    last_batch_policy = DROP, last_batch_padded = False  -> last batch = ``[5, 6]``, next iteration will return ``[2, 3]``
    """
    def __init__(self,
                 pipelines,
                 output_map,
                 size=-1,
                 reader_name=None,
                 auto_reset=True,
                 fill_last_batch=None,
                 dynamic_shape=False,
                 last_batch_padded=False,
                 last_batch_policy=LastBatchPolicy.FILL,
                 prepare_first_batch=True):

        # check the assert first as _DaliBaseIterator would run the prefetch
        assert len(set(output_map)) == len(output_map), "output_map names should be distinct"
        self._output_categories = set(output_map)
        self.output_map = output_map

        _DaliBaseIterator.__init__(self,
                                   pipelines,
                                   size,
                                   reader_name,
                                   auto_reset,
                                   fill_last_batch,
                                   last_batch_padded,
                                   last_batch_policy,
                                   prepare_first_batch=prepare_first_batch)
        self._dynamic_shape = dynamic_shape

        # Use double-buffering of data batches
        self._data_batches = [None for i in range(self._num_gpus)]

        self._first_batch = None
        if self._prepare_first_batch:
            try:
                self._first_batch = DALIGenericIterator.__next__(self)
            except StopIteration:
                assert False, "It seems that there is no data in the pipeline. This may happen if `last_batch_policy` is set to PARTIAL and the requested batch size is greater than the shard size."

    def __next__(self):
        if self._first_batch is not None:
            batch = self._first_batch
            self._first_batch = None
            return batch

        # Gather outputs
        outputs = self._get_outputs()
        for i in range(self._num_gpus):
            dev_id = self._pipes[i].device_id

            #images, targets, offset
            out_images = []
            targets = []
            labels = []

            # segregate outputs into categories
            for j, out in enumerate(outputs[i]):
                if self.output_map[j] == "image":
                    out_images.append(out)
                elif self.output_map[j] == "targets":
                    targets.append(out)
                elif self.output_map[j] == "labels":
                    labels.append(out)

            # Change DALI TensorLists into Tensors
            images = [x.as_tensor() for x in out_images]
            images_shape = [x.shape() for x in images]

            # Prepare bboxes shapes
            targets_shape = []
            for j in range(len(targets)):
                targets_shape.append([])
                for k in range(len(targets[j])):
                    targets_shape[j].append(targets[j][k].shape())

            # Prepare labels shapes and offsets
            target_offsets = []   
            torch.cuda.synchronize()
            for j in range(len(labels)):
                target_offsets.append([0])
                for k in range(len(labels[j])):
                    lshape = labels[j][k].shape()
                    target_offsets[j].append(target_offsets[j][k] + lshape[0])

            # We always need to alocate new memory as bboxes and labels varies in shape
            images_torch_type = to_torch_type[np.dtype(images[0].dtype())]
            targets_torch_type = to_torch_type[np.dtype(targets[0][0].dtype())]

            torch_gpu_device = torch.device('cuda', dev_id)
            torch_cpu_device = torch.device('cpu')

            pyt_images = [torch.zeros(shape, dtype=images_torch_type, device=torch_gpu_device) for shape in images_shape]
            pyt_targets = [[torch.zeros(shape, dtype=targets_torch_type, device=torch_gpu_device) for shape in shape_list] for shape_list in targets_shape]
            pyt_offsets = [torch.zeros(len(offset), dtype=torch.int32, device=torch_cpu_device) for offset in target_offsets]
            self._data_batches[i] = (pyt_images, pyt_targets, pyt_offsets)

            # Copy data from DALI Tensors to torch tensors
            for j, i_arr in enumerate(images):
                feed_ndarray(i_arr, pyt_images[j])

            for j, b_list in enumerate(targets):
                for k in range(len(b_list)):
                    if (pyt_targets[j][k].shape[0] != 0):
                        feed_ndarray(b_list[k], pyt_targets[j][k])
                pyt_targets[j] = torch.cat(pyt_targets[j])

            for j in range(len(pyt_offsets)):
                pyt_offsets[j] = torch.IntTensor(target_offsets[j])


        self._schedule_runs()

        self._advance_and_check_drop_last()

        if self._reader_name:
            if_drop, left = self._remove_padded()
            if np.any(if_drop):
                output = []
                for batch, to_copy in zip(self._data_batches, left):
                    batch = batch.copy()
                    for category in self._output_categories:
                        batch[category] = batch[category][0:to_copy]
                    output.append(batch)
                return output

        else:
            if self._last_batch_policy == LastBatchPolicy.PARTIAL and (self._counter > self._size) and self._size > 0:
                # First calculate how much data is required to return exactly self._size entries.
                diff = self._num_gpus * self.batch_size - (self._counter - self._size)
                # Figure out how many GPUs to grab from.
                numGPUs_tograb = int(np.ceil(diff/self.batch_size))
                # Figure out how many results to grab from the last GPU (as a fractional GPU batch may be required to
                # bring us right up to self._size).
                mod_diff = diff % self.batch_size
                data_fromlastGPU = mod_diff if mod_diff else self.batch_size

                # Grab the relevant data.
                # 1) Grab everything from the relevant GPUs.
                # 2) Grab the right data from the last GPU.
                # 3) Append data together correctly and return.
                output = self._data_batches[0:numGPUs_tograb]
                output[-1] = output[-1].copy()
                for category in self._output_categories:
                    output[-1][category] = output[-1][category][0:data_fromlastGPU]
                return output

        return self._data_batches

class DaliWiderfaceDataset(object):
    def __init__(self,
                 pipelines,
                 output_map,
                 size=-1,
                 reader_name=None,
                 auto_reset=True,
                 fill_last_batch=None,
                 dynamic_shape=False,
                 last_batch_padded=False,
                 last_batch_policy=LastBatchPolicy.FILL,
                 prepare_first_batch=True):
        super().__init__()
        self.dataloader = DALIGenericIterator(
                            pipelines,
                            output_map,
                            size=size,
                            reader_name=reader_name,
                            auto_reset=auto_reset,
                            fill_last_batch=fill_last_batch,
                            dynamic_shape=dynamic_shape,
                            last_batch_padded=last_batch_padded,
                            last_batch_policy=last_batch_policy,
                            prepare_first_batch=prepare_first_batch)
     
    def _dali_collate(self, input):
        pyt_images, pyt_targets ,pyt_offsets = input[0]
        images, targets, offsets = pyt_images[0], pyt_targets[0], pyt_offsets[0]
        label = torch.ones(targets.shape[0]).view(-1, 1).to(images.get_device())
        targets_tensor = torch.cat([targets, label], dim=1)
        lastid = 0
        targets = []
        for ost in offsets[1:]:
            targets.append(targets_tensor[lastid: ost])
            lastid = ost
        images = images.float().permute(0, 3, 1, 2).contiguous()  
        return images, targets 

    def __iter__(self):
        return self
    
    def __next__(self):
        return self._dali_collate(next(self.dataloader))
    
    def __len__(self):
        return len(self.dataloader)

def get_train_loader(imgs_root, annos_file, local_seed = -1, num_gpus = 1, batch_size = 1, num_workers = 1, device_id = 0, shuffle=True, shuffle_after_epoch=False):
    train_pipe = WiderfacePipeline_COCOformat(file_root=imgs_root, 
                                annotations_file=annos_file, 
                                batch_size = batch_size, 
                                num_threads = num_workers,
                                device_id = device_id, 
                                seed = local_seed, 
                                num_gpus=num_gpus,
                                random_shuffle=shuffle,
                                shuffle_after_epoch=shuffle_after_epoch
                                )        
    train_loader = DaliWiderfaceDataset(
                            train_pipe,
                            output_map = ["image", "targets", "labels"],
                            reader_name="Reader",
                            last_batch_policy=LastBatchPolicy.FILL)
    return train_loader
