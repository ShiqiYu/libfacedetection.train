#-*- coding:utf-8 -*-
import os
from typing import List
import torch
import numpy as np

#import for dali dataloader
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
from nvidia.dali import types
from nvidia.dali import math as math
from nvidia.dali.plugin.base_iterator import _DaliBaseIterator
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
import ctypes

import cv2
import scipy
import json
import torch.utils.data as data

class WiderfacePipeline_COCOformat(Pipeline):
    def __init__(self, batch_size, device_id, 
                file_root, annotations_file, num_gpus, input_dim=320.,
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
        box_like_shape = fn.cat(fn.slice(fn.shapes(bboxes, dtype=types.INT32), 0, 1, axes = [0]), -1) 
        targets = fn.cat(bboxes, fn.reshape(vertices, shape=box_like_shape), axis=1)

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
                imgs_root, 
                annos_file, 
                local_seed = -1, 
                num_gpus = 1, 
                batch_size = 1, 
                num_workers = 1, 
                device_id = 0, 
                shuffle=True, 
                shuffle_after_epoch=False, 
                img_dim=320.,
                ) -> None:
        super().__init__()
        if isinstance(img_dim, List):
            assert len(img_dim) == 2 and img_dim[1] > img_dim[0]
            self.multi_scale = True
        else:
            self.multi_scale = False
        self.img_dim = img_dim
        self.imgs_root = imgs_root 
        self.annos_file = annos_file
        self.local_seed = local_seed 
        self.num_gpus = num_gpus
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device_id = device_id
        self.shuffle = shuffle
        self.shuffle_after_epoch = shuffle_after_epoch
        self.output_map=["image", "targets", "labels"]
        self.size=-1
        self.reader_name="Reader"
        self.auto_reset=True
        self.fill_last_batch=None
        self.dynamic_shape=False
        self.last_batch_padded=False
        self.last_batch_policy=LastBatchPolicy.FILL
        self.prepare_first_batch=True
        if self.multi_scale:
            self._init_dataloader(self.img_dim[1])
        else:
            self._init_dataloader(self.img_dim)

    def _init_dataloader(self, img_dim):
        train_pipe = WiderfacePipeline_COCOformat(
                            file_root=self.imgs_root, 
                            annotations_file=self.annos_file, 
                            batch_size = self.batch_size, 
                            num_threads = self.num_workers,
                            device_id = self.device_id, 
                            seed = self.local_seed, 
                            num_gpus=self.num_gpus,
                            random_shuffle=self.shuffle,
                            shuffle_after_epoch=self.shuffle_after_epoch,
                            input_dim=img_dim
                            ) 
        self.dataloader = DALIGenericIterator(
                            train_pipe,
                            output_map=self.output_map,
                            size=self.size,
                            reader_name=self.reader_name,
                            auto_reset=self.auto_reset,
                            fill_last_batch=self.fill_last_batch,
                            dynamic_shape=self.dynamic_shape,
                            last_batch_padded=self.last_batch_padded,
                            last_batch_policy=self.last_batch_policy,
                            prepare_first_batch=self.prepare_first_batch)

    def reset(self):
        if not self.multi_scale:
            return
        else:
            img_dim = np.random.randint(self.img_dim[0], self.img_dim[1] + 1)
            self._init_dataloader(img_dim)
    
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

class WIDERFace(data.Dataset):
    '''Dataset class for WIDER Face dataset'''

    def __init__(self, kargs):
        self.root = kargs.get('root', None)
        self.split = kargs.get('split', 'val')
        assert self.root is not None

        self.widerface_img_paths = {
            'val':  os.path.join(self.root, 'WIDER_val', 'images'),
            'test': os.path.join(self.root, 'WIDER_test', 'images')
        }

        self.widerface_split_fpaths = {
            'val':  os.path.join(self.root, 'wider_face_split', 'wider_face_val.mat'),
            'test': os.path.join(self.root, 'wider_face_split', 'wider_face_test.mat')
        }

        self.img_list, self.num_img = self.load_list()

    def load_list(self):
        n_imgs = 0
        flist = []

        split_fpath = self.widerface_split_fpaths[self.split]
        img_path = self.widerface_img_paths[self.split]

        anno_data = scipy.io.loadmat(split_fpath)
        event_list = anno_data.get('event_list')
        file_list = anno_data.get('file_list')

        for event_idx, event in enumerate(event_list):
            event_name = event[0][ 0]
            for f_idx, f in enumerate(file_list[event_idx][0]):
                f_name = f[0][0]
                f_path = os.path.join(img_path, event_name, f_name+'.jpg')
                flist.append(f_path)
                n_imgs += 1

        return flist, n_imgs

    def __getitem__(self, index):
        img = cv2.imread(self.img_list[index])
        event, name = self.img_list[index].split('/')[-2:]
        mata = {}
        mata['event'] = event
        mata['name'] = name
        return img, mata

    def __len__(self):
        return self.num_img

    @property
    def size(self):
        return self.num_img



class CCPDtestloader(object):
    def __init__(self, kargs) -> None:
        super().__init__()
        self.root = kargs.get('root', None)
        self.split = kargs.get('split', None)
        assert self.root == None or self.root == None
        annos_file = os.path.join(self.root, "splits_coco", f"{self.split}.json")
        with open(annos_file, "r") as f:
            annos = json.load(f)
        self.images = []
        self.matas = []
        for img in annos['images']:
            self.images.append(os.path.join(self.root, img["file_name"]))
            self.matas.append({'id': img['id']})

    def __getitem__(self, index):
        img = cv2.imread(self.images[index])
        mata = self.matas[index]
        return img, mata
    
    def __len__(self):
        return len(self.images)

class HanCotestloader(object):
    def __init__(self, kargs) -> None:
        super().__init__()
        self.root = kargs.get('root', None)
        self.split = kargs.get('split', None)
        assert self.root == None or self.root == None
        annos_file = os.path.join(self.root, "detection_merge", f"{self.split}.json")
        with open(annos_file, "r") as f:
            annos = json.load(f)
        self.images = []
        self.matas = []
        for img in annos['images']:
            self.images.append(os.path.join(self.root, img["file_name"]))
            self.matas.append({'id': img['id']})

    def __getitem__(self, index):
        img = cv2.imread(self.images[index])
        mata = self.matas[index]
        return img, mata
    
    def __len__(self):
        return len(self.images)





class widerface_mosaic(data.Dataset):
    def __init__(self, 
                root,
                annotations_file,
                use_mosaic,  
                size  
        ) -> None:
        super().__init__()
        self.use_mosaic = use_mosaic

        self.size = size
        self.num_landmarks = 5
        self.img_matas = self.read_coco(root, annotations_file)
        self.n = len(self.img_matas)

    def __len__(self):
        return self.n
    
    def __getitem__(self, index):
        if self.use_mosaic:
            img, target = self.load_mosaic_face(size=self.size, index=index)
        else:
            img, target = self.load_crop_face(size=self.size, index=index)
        return img, target
    
    def safe_resize_flip(self, dsize_w, dsize_h, image, boxes, isflip):
        h, w, c = image.shape
        r_w, r_h = dsize_w / w, dsize_h / h
        r = r_w
        if r < r_h:
            r = r_h
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            image = cv2.resize(image, (int(w * r), dsize_h), interpolation=interp)
        elif r == r_h:
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            image = cv2.resize(image, (dsize_w, dsize_h), interpolation=interp)
        else:
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            image = cv2.resize(image, (dsize_w, int(h * r)), interpolation=interp)
        
        boxes *= r 
        # flip
        # if isflip and np.random.uniform(0., 1.) > 0.5:
        #     h, w, c = image.shape
        #     image = cv2.flip(image, 1)
        #     boxes[:, 4::2] = w - boxes[:, 4::2][:, ::-1]       

        return image
            
    def load_mosaic_face(self, size, index, isflip=True):
        indices = [index] + [np.random.randint(0, self.n) for _ in range(3)]# 3 additional image indices
        yc, xc = np.random.uniform(0.25 * size, 0.75 * size, 2) # mosaic center x, y
        yc = int(yc)
        xc = int(xc)

        boxes_all = np.empty((0, 2 * self.num_landmarks + 4))
        labels_all = np.empty((0,))
        for i in range(4):
            img_mata = self.img_matas[indices[i]]
            image = cv2.imread(img_mata['image_path'])
            boxes = img_mata['annotations'].copy()
            labels = img_mata['labels'].copy()


            if i == 0: # top left
                img4 = np.full((size, size, 3), 0, dtype=np.uint8)  # base image with 4 tiles

                image = self.safe_resize_flip(xc, yc, image, boxes, isflip)
                h, w, _ = image.shape             
                x1a, y1a, x2a, y2a = 0, 0, xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - xc, h - yc, w, h  # xmin, ymin, xmax, ymax (small image)
              
            elif i == 1:  # top right
                image = self.safe_resize_flip(size - xc, 0, image, boxes, isflip)
                h, w, _ = image.shape           
                h_right = min(int(0.75 * size), h)
                x1a, y1a, x2a, y2a = xc, 0, min(xc + w, size), h_right
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), w, h

            elif i == 2:  # bottom left
                image = self.safe_resize_flip(xc, size - yc, image, boxes, isflip)
                h, w, _ = image.shape          
                x1a, y1a, x2a, y2a = 0, yc, xc, size
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, y2a - y1a
                                  
            elif i == 3:  # bottom right
                image = self.safe_resize_flip(size - xc, size - h_right, image, boxes, isflip)   
                h, w, _ = image.shape             
                x1a, y1a, x2a, y2a = xc, h_right, size, size
                x1b, y1b, x2b, y2b = 0, 0, x2a - x1a, y2a - y1a

                    
            img4[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            boxes[:, 0::2] += x1a - x1b
            boxes[:, 1::2] += y1a - y1b
            boxes_all = np.concatenate([boxes_all, boxes], axis=0)
            labels_all = np.concatenate([labels_all, labels], axis=0)
        center = (boxes_all[:, 0:2] + boxes_all[:, 2:4]) / 2
        mask = np.logical_and((center > 0), center < size).all(axis=1)
        boxes_all = boxes_all[mask]
        boxes_all.clip(0, size)
        labels_all = labels_all[mask]
        boxes_all[:, :] /= size
        target = np.concatenate([boxes_all, labels_all[:, None]], axis=-1)

        return img4, target

    def load_crop_face(self, index, size, isflip=True):
        img_mata = self.img_matas[index]
        image = cv2.imread(img_mata['image_path'])
        boxes = img_mata['annotations'].copy()
        labels = img_mata['labels'].copy()
        height, width, _ = image.shape

        # # flip
        # if isflip and np.random.uniform(0., 1.) > 0.5:
        #     image = cv2.flip(image, 1)
        #     boxes[:, 0::2] = width - boxes[:, 0::2]

        # crop
        attemp_num = 1024
        scale = [0.3, 1.]
        crop_size = np.random.uniform(*scale) * min(height, width)
        w_anchor, h_anchor = width - crop_size, height - crop_size

        for _ in range(attemp_num):
            xmin = w_anchor * np.random.uniform(0., 1.)
            ymin = h_anchor * np.random.uniform(0., 1.)
            xmax = xmin + crop_size
            ymax = ymin + crop_size
            roi = np.array([xmin, ymin, xmax, ymax]).astype(np.int32)

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

            image = image_t
            boxes = boxes_t
            labels = labels_t
            attemp_success = True
            break
        
        if not attemp_success:
            long_side = max(width, height)
            image_t = np.empty((long_side, long_side, 3), dtype=image.dtype)
            image_t[0:0 + height, 0:0 + width] = image
            image = image_t
        
        # normalize
        height, width, _ = image.shape
        boxes[:, 0::2] /= width
        boxes[:, 1::2] /= height

        # resize
        interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        interp_method = interp_methods[np.random.randint(0, len(interp_methods))]
        image = cv2.resize(image, (size, size), interpolation=interp_method)

        image = image.astype(np.float32)
        target = np.concatenate([boxes, labels[:, None]], axis=-1, dtype=np.float32)

        return image, target

    def collate_fn(self, batch):
        targets = []
        images = []
        for sample in batch:
            image, target = sample
            images.append(torch.from_numpy(image).permute(2, 0, 1))
            targets.append(torch.from_numpy(target))
        
        return (torch.stack(images, 0).contiguous().float(), targets)

    def read_coco(self, root, annotations_file):
        assert os.path.exists(annotations_file)
        assert os.path.exists(root)
        with open(annotations_file, 'r') as f:
            targets = json.load(f)
        img_matas = []
        imgs_dict = targets['images']
        annos_dict = targets['annotations']
        anno_id = 0
        anno_num = len(annos_dict)
        for per_img in imgs_dict:
            img_id = per_img['id']
            img_path = os.path.join(root, per_img['file_name'])
            labels = []
            annos = np.empty((0, self.num_landmarks * 2 + 4), dtype=np.float32)
            for i in range(anno_id, anno_num):
                if img_id == annos_dict[i]['image_id']:
                    bbox = annos_dict[i]['bbox']
                    bbox[2] += bbox[0]
                    bbox[3] += bbox[1]
                    landmarks = annos_dict[i]['segmentation'][0]
                    annos = np.concatenate([annos, np.array(bbox + landmarks, dtype=np.float32).reshape(1, -1)], axis=0)
                    labels.append(1)
                else:
                    anno_id = i
                    break
            labels = np.array(labels, dtype=np.int32)
            img_matas.append({'image_path': img_path, "annotations": annos, "labels": labels})
        return img_matas

class Widerface_pytorch_loader(object):
    def __init__(self, 
                root, 
                annotations_file, 
                num_workers,
                batch_size, 
                out_sizes,
                pin_memory=True, 
                shuffle=True, 
                seed=347
        ) -> None:
        super().__init__()
        assert isinstance(out_sizes, List) and len(out_sizes) == 2
        self.out_sizes = out_sizes
        self.root=root
        self.annotations_file=annotations_file
        self.num_workers=num_workers
        self.pin_memory=pin_memory
        self.shuffle=shuffle
        self.batch_size = batch_size
        np.random.seed(seed)
        self.reset(size=out_sizes[1])

    def reset(self, size=0, use_mosaic=True):
        if size == 0:
            self.size = np.random.randint(self.out_sizes[0], self.out_sizes[1] + 1)
        else:
            self.size = size
        dataset = widerface_mosaic(
                root=self.root,
                annotations_file=self.annotations_file,
                use_mosaic=use_mosaic,
                size=self.size
        )
        self.dataloader = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=dataset.collate_fn,
                shuffle=self.shuffle
        )
    
    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        return iter(self.dataloader)
    

if __name__ == '__main__':
    # loader = Widerface_pytorch_loader(
    #     root='/home/ww/projects/yudet/data/widerface/WIDER_train/images',
    #     annotations_file='/home/ww/projects/yudet/data/widerface/trainset.json',
    #     batch_size=16,
    #     num_workers=1,
    #     pin_memory=True,
    #     out_sizes=[320, 640]
    # )

    # for ids, one_batch_data in enumerate(loader):
    #     images, targets = one_batch_data
    #     images = images.permute(0, 2, 3, 1).contiguous()
    #     images = images.numpy().astype(np.uint8)
    #     for idx in range(images.shape[0]):
    #         target, image = targets[idx], images[idx]
    #         h, w, c = image.shape
    #         bboxs = (target[:, :14] * w).numpy().astype(np.int32)
    #         for box in bboxs:
    #             x1, y1, x2, y2 = box[:4]
    #             ldm = box[4:]
    #             cv2.rectangle(image, pt1=(int(x1), int(y1)), pt2=(int(x2), int(y2)), color=(255, 255, 7))
    #             for s in range(0, len(ldm), 2):
    #                 cv2.circle(image, center=(int(ldm[s]), int(ldm[s+1])), radius=1, color=(7, 255, 255))
    #         cv2.imwrite(f"./images/woca_{ids:04d}_{idx:04d}.jpg", image)
    #     if ids > 50:
    #         break
    dataset = widerface_mosaic(
            root='/home/ww/projects/yudet/data/widerface/WIDER_train/images',
            annotations_file='/home/ww/projects/yudet/data/widerface/trainset.json',
            use_mosaic=True,
            size=640
        )
    np.random.seed(347)
    for i in range(len(dataset)):
        image, target = dataset.__getitem__(i)
        h, w, c = image.shape
        bboxs = (target[:, :14] * w).astype(np.int32)
        for box in bboxs:
            x1, y1, x2, y2 = box[:4]
            ldm = box[4:]
            cv2.rectangle(image, pt1=(int(x1), int(y1)), pt2=(int(x2), int(y2)), color=(255, 255, 7))
            for s in range(0, len(ldm), 2):
                cv2.circle(image, center=(int(ldm[s]), int(ldm[s+1])), radius=1, color=(7, 255, 255))
            cv2.imwrite(f"./images/woca_{i:04d}.jpg", image)
