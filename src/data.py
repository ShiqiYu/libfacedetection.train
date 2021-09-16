#-*- coding:utf-8 -*-
import os
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
