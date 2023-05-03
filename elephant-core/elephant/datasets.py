# Copyright (c) 2020, Ko Sugawara
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1.  Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
# 2.  Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ==============================================================================
"""Implementations of PyTorch dataset modules."""

import math
import os
from pathlib import Path
from random import uniform
from random import randint
from random import randrange

from filelock import FileLock
import numpy as np
from skimage.transform import rotate
import torch
import torch.nn.functional as F
import torch.utils.data as du
if os.environ.get('CTC') != '1':
    import zarr

from elephant.logging import logger
from elephant.util import LRUCacheDict
from elephant.util import normalize_zero_one
from elephant.util import RUN_ON_FLASK
from elephant.redis_util import get_state
from elephant.redis_util import redis_client
from elephant.redis_util import REDIS_KEY_NCROPS
from elephant.redis_util import REDIS_KEY_TIMEPOINT
from elephant.redis_util import TrainState

PROFILE = "ELEPHANT_PROFILE" in os.environ

if os.environ.get('ELEPHANT_PROFILE') is None:
    def profile(func):
        return func


def _load_image(za_input, timepoint, use_median=False, img_size=None):
    img = za_input[timepoint].astype('float32')
    if use_median and img.ndim == 3:
        global_median = np.median(img)
        for z in range(img.shape[0]):
            slice_median = np.median(img[z])
            if 0 < slice_median:
                img[z] -= slice_median - global_median
    img = normalize_zero_one(img)
    if img_size is not None:
        img = F.interpolate(
            torch.from_numpy(img)[None, None],
            size=img_size,
            mode='trilinear' if img.ndim == 3 else 'bilinear',
            align_corners=True,
        )[0, 0].numpy()
    return img


def _get_memmap_or_load(za, timepoint, memmap_dir=None, use_median=False,
                        img_size=None):
    if memmap_dir:
        key = f'{Path(za.store.path).parent.name}-t{timepoint}-{use_median}'
        fpath_org = Path(memmap_dir) / f'{key}.dat'
        if img_size is not None:
            key += '-' + '-'.join(map(str, img_size))
        fpath = Path(memmap_dir) / f'{key}.dat'
        lock = FileLock(str(fpath) + '.lock')
        with lock:
            if not fpath.exists():
                logger().info(f'creating {fpath}')
                fpath.parent.mkdir(parents=True, exist_ok=True)
                img_org = np.memmap(
                    fpath_org,
                    dtype='float32',
                    mode='w+',
                    shape=za.shape[1:]
                )
                img_org[:] = za[timepoint].astype('float32')
                if img_size is None:
                    img = img_org
                else:
                    img = np.memmap(
                        fpath,
                        dtype='float32',
                        mode='w+',
                        shape=img_size
                    )
                    img[:] = F.interpolate(
                        torch.from_numpy(img_org)[None, None],
                        size=img_size,
                        mode='trilinear' if img.ndim == 3 else 'bilinear',
                        align_corners=True,
                    )[0, 0].numpy()
                if use_median and img.ndim == 3:
                    global_median = np.median(img)
                    for z in range(img.shape[0]):
                        slice_median = np.median(img[z])
                        if 0 < slice_median:
                            img[z] -= slice_median - global_median
                img = normalize_zero_one(img)
            logger().info(f'loading from {fpath}')
            return np.memmap(
                fpath,
                dtype='float32',
                mode='c',
                shape=za.shape[1:] if img_size is None else img_size
            )
    else:
        img = za[timepoint].astype('float32')
        if use_median and img.ndim == 3:
            global_median = np.median(img)
            for z in range(img.shape[0]):
                slice_median = np.median(img[z])
                if 0 < slice_median:
                    img[z] -= slice_median - global_median
        img = normalize_zero_one(img)
    return img


def get_input_at(za_input, timepoint, cache_dict=None, memmap_dir=None,
                 use_median=False, img_size=None):
    if cache_dict:
        key = f'{za_input.store.path}-t{timepoint}-{use_median}'
        if img_size is not None:
            key += '-' + '-'.join(map(str, img_size))
        cache = cache_dict.get(key)
        if cache is None:
            cache = cache_dict.get(
                key,
                _get_memmap_or_load(za_input, timepoint, memmap_dir,
                                    use_median, img_size)
            )
        return cache
    return _get_memmap_or_load(za_input, timepoint, memmap_dir, use_median,
                               img_size)


def get_inputs_at(za_input, timepoint, cache_dict=None, memmap_dir=None,
                  img_size=None):
    if cache_dict:
        key = f'{za_input.store.path}-t{timepoint}-t{timepoint+1}'
        if img_size is not None:
            key += '-' + '-'.join(map(str, img_size))
        cache = cache_dict.get(key)
        if cache is None:
            cache = cache_dict.get(
                key,
                np.array([get_input_at(za_input,
                                       i,
                                       cache_dict,
                                       memmap_dir,
                                       img_size=img_size)
                          for i in (timepoint, timepoint+1)])
            )
        return cache
    return np.array([get_input_at(za_input,
                                  i,
                                  cache_dict,
                                  memmap_dir,
                                  img_size=img_size)
                     for i in (timepoint, timepoint+1)])


class DatasetPrediction(du.Dataset):
    def __init__(self, input, patch_list, keep_axials):
        """
        input: 5D tensor
        """
        self.input = input
        self.patch_list = patch_list
        self.keep_axials = keep_axials

    def __len__(self):
        return len(self.input) * len(self.patch_list)

    def __getitem__(self, index):
        if (redis_client is not None and
                get_state() == TrainState.IDLE.value):
            raise KeyboardInterrupt
        data_ind = index // len(self.patch_list)
        patch_ind = index % len(self.patch_list)
        slices, _ = self.patch_list[patch_ind]
        return (self.input[(data_ind,) + tuple(slices)],
                self.keep_axials[data_ind], data_ind, patch_ind)


class SegmentationDatasetBase(du.Dataset):
    def __init__(self, crop_size, keep_axials=(True,) * 4, scales=None,
                 scale_factor_base=0, rotation_angle=None, contrast=0.5,
                 is_eval=False):
        """Generate dataset for segmentation.

        Args:
            crop_size(array-like of length ndim): crop size to generate dataset.
            keep_axials(array-like of length 4): this value is used to calculate
                how many times down/up sampling are performed in z direction.
                Ignored for 2D data.
            scales(array-like of length ndim): a list of pixel/voxel size in
                physical unit (e.g. 0.5 μm/px). This is used to calculate scale
                factors for augmentation.
            scale_factor_base(float): a base scale factor for augmentation.
            rotation_angle(float): rotation angle for augmentation in degree.
            contrast(float): contrast factor for augmentation.
            is_eval(boolean): True if the dataset is for evaluation, where no
                augmentation is performed.
        """
        if scale_factor_base < 0 or 1 <= scale_factor_base:
            raise ValueError(
                'scale_factor_base should be 0 <= scale_factor_base < 1'
            )
        self.n_dims = len(crop_size)
        if scales is None:
            scales = (1.,) * self.n_dims
        scale_factors = tuple(
            scale_factor_base * min(scales) / scales[i]
            for i in range(self.n_dims)
        )
        self.crop_size = crop_size
        self.scale_factors = scale_factors
        self.rotation_angle = rotation_angle
        self.contrast = contrast
        self.is_eval = is_eval
        self.keep_axials = torch.tensor(keep_axials)

    def _generate_item(self, img_input, img_label, crop_size):
        if self.is_eval:
            tensor_input = torch.from_numpy(img_input[None])
            tensor_label = torch.from_numpy(img_label).long()
            return (tensor_input, self.keep_axials), tensor_label - 1

        while True:
            if 0 < sum(self.scale_factors):
                item_crop_size = [
                    randrange(
                        min(
                            img_input.shape[i],
                            round(crop_size[i] *
                                  (1. - self.scale_factors[i]))
                        ),
                        min(
                            img_input.shape[i] + 1,
                            int(crop_size[i] *
                                (1. + self.scale_factors[i])) + 1
                        )
                    ) for i in range(self.n_dims)
                ]
            else:
                item_crop_size = crop_size

            if self.rotation_angle is not None and 0 < self.rotation_angle:
                # rotate image
                theta = randint(-self.rotation_angle, self.rotation_angle)
                cos_theta = math.cos(math.radians(theta))
                sin_theta = math.sin(math.radians(theta))
                for i in (-2, -1):
                    item_crop_size[i] *= (abs(cos_theta) + abs(sin_theta))
                    item_crop_size[i] = math.ceil(item_crop_size[i])
                item_crop_size = [
                    min(img_input.shape[i], item_crop_size[i])
                    for i in range(self.n_dims)
                ]

            if isinstance(self, SegmentationDatasetZarr):
                if not self.is_ae:
                    za_label_a = zarr.open(self.zpath_seg_label, mode='a')
                    index_pool = za_label_a.attrs.get(
                        f'label.indices.{self.i_frame}')
                    if index_pool is None:
                        index_pool = np.argwhere(0 < img_label)
                        za_label_a.attrs[
                            f'label.indices.{self.i_frame}'
                        ] = tuple(map(tuple, index_pool.tolist()))
            else:
                index_pool = np.argwhere(0 < img_label)
            if self.is_ae:
                origins = [
                    randint(0, img_input.shape[i] - item_crop_size[i])
                    for i in range(self.n_dims)
                ]
            else:
                base_index = index_pool[randrange(len(index_pool))]
                origins = [
                    randint(
                        max(0,
                            (int(base_index[i] * self.resize_factor[i]) -
                             (item_crop_size[i] - 1))),
                        min((img_input.shape[i] - item_crop_size[i]),
                            int(base_index[i] * self.resize_factor[i]))
                    )
                    for i in range(self.n_dims)
                ]
            slices = tuple(
                slice(origins[i], origins[i] + item_crop_size[i])
                for i in range(self.n_dims)
            )
            if not self.is_ae:
                sliced_label = img_label[slices].copy()
            sliced_input = img_input[slices].copy()

            if not self.is_ae and 0 < self.contrast:
                fg_index = np.isin(sliced_label, (2, 3, 5, 6))
                bg_index = np.isin(sliced_label, (1, 4))
                if fg_index.any() and bg_index.any():
                    fg_mean = sliced_input[fg_index].mean()
                    bg_mean = sliced_input[bg_index].mean()
                    cr_factor = (
                        ((fg_mean - bg_mean) * uniform(self.contrast, 1)
                         + bg_mean)
                        / fg_mean
                    )
                    sliced_input[fg_index] *= cr_factor

            if self.rotation_angle is not None and 0 < self.rotation_angle:
                if self.n_dims == 3:
                    sliced_input = np.array([
                        rotate(
                            sliced_input[z],
                            theta,
                            resize=True,
                            preserve_range=True,
                            order=1,  # 1: Bi-linear (default)
                        ) for z in range(sliced_input.shape[0])
                    ])
                else:
                    sliced_input = rotate(sliced_input,
                                          theta,
                                          resize=True,
                                          preserve_range=True,
                                          order=1,  # 1: Bi-linear (default)
                                          )
                h_crop, w_crop = crop_size[-2:]
                h_rotate, w_rotate = sliced_input.shape[-2:]
                r_origin = max(0, (h_rotate - h_crop) // 2)
                c_origin = max(0, (w_rotate - w_crop) // 2)
                sliced_input = sliced_input[...,
                                            r_origin:r_origin+h_crop,
                                            c_origin:c_origin+w_crop]

                # rotate label
                if not self.is_ae:
                    if self.n_dims == 3:
                        sliced_label = np.array([
                            rotate(
                                sliced_label[z],
                                theta,
                                resize=True,
                                preserve_range=True,
                                order=0,  # 0: Nearest-neighbor
                            ) for z in range(sliced_label.shape[0])
                        ])
                    else:
                        sliced_label = rotate(sliced_label,
                                              theta,
                                              resize=True,
                                              preserve_range=True,
                                              order=0,  # 0: Nearest-neighbor
                                              )
                    sliced_label = sliced_label[...,
                                                r_origin:r_origin+h_crop,
                                                c_origin:c_origin+w_crop]
                    if sliced_label.max() == 0:
                        continue
            break

        tensor_input = torch.from_numpy(sliced_input)[None]
        if not self.is_ae:
            tensor_label = torch.from_numpy(sliced_label)
        if tensor_input.shape[1:] != crop_size:
            interpolate_mode = 'trilinear' if self.n_dims == 3 else 'bilinear'
            tensor_input = F.interpolate(tensor_input[None],
                                         crop_size,
                                         mode=interpolate_mode,
                                         align_corners=True)[0]
            if not self.is_ae:
                tensor_label = F.interpolate(
                    tensor_label[None, None].float(),
                    crop_size,
                    mode='nearest'
                )[0, 0]
        if self.is_ae:
            return (tensor_input, self.keep_axials), tensor_input
        tensor_label = tensor_label.long()
        flip_dims = [-(1 + i)
                     for i, v in enumerate(torch.rand(self.n_dims)) if v < 0.5]
        tensor_input = torch.flip(tensor_input, flip_dims)
        tensor_label = torch.flip(tensor_label, flip_dims)
        # -1: unlabeled, 0: BG (LW), 1: Outer (LW), 2: Inner (LW)
        #                3: BG (HW), 4: Outer (HW), 5: Inner (HW)
        return (tensor_input, self.keep_axials), tensor_label - 1


class SegmentationDatasetZarr(SegmentationDatasetBase):
    def __init__(self, zpath_input, zpath_seg_label, indices, img_size,
                 crop_size, n_crops, keep_axials=(True,) * 4, scales=None,
                 is_livemode=False, scale_factor_base=0.2, is_ae=False,
                 rotation_angle=None, contrast=0.5, is_eval=False, length=None,
                 adaptive_length=False, cache_maxbytes=None, memmap_dir=None):
        """Generate dataset for segmentation.

        Args:
            zpath_input(String): a path to .zarr file for input data.
            zpath_seg_label(String): a path to .zarr file for label data.
            indices(list): a list of timepoints to be used.
            img_size(array-like of length ndim): input image size.
            crop_size(array-like of length ndim): crop size to generate dataset.
            n_crops(int): number of crops per timepoint.
            keep_axials(array-like of length 4): this value is used to calculate
                how many times down/up sampling are performed in z direction.
                Ignored for 2D data.
            scales(array-like of length ndim): a list of pixel/voxel size in
                physical unit (e.g. 0.5 μm/px). This is used to calculate scale
                factors for augmentation.
            is_livemode(boolean): True if training is performed in live mode.
            scale_factor_base(float): a base scale factor for augmentation.
            is_ae(boolean): True if called from a prior training.
            rotation_angle(float): rotation angle for augmentation in degree.
            contrast(float): contrast factor for augmentation.
            is_eval(boolean): True if the dataset is for evaluation, where no
                augmentation is performed.
            length(int): a lenght of this dataset. If None, it is automatically
                calculated by the length of indices and n_crops.
            adaptive_length(boolean): True if the length of the dataset is
                adjusted adaptively. For example, given that the length is 10
                and the len(indices) * self.n_crops is 6, the length becomes 6
                if adaptive_length is true, while it remains 10 if false.
            cache_maxbytes (int): size of the memory capacity for cache in byte.
            memmap_dir (str): path to a directory for storing memmap files.
        """
        if len(img_size) != len(crop_size):
            raise ValueError(
                'img_size: {} and crop_size: {} should have the same length'
                .format(img_size, crop_size)
            )
        super().__init__(crop_size, keep_axials, scales, scale_factor_base,
                         rotation_angle, contrast, is_eval)

        self.za_input = zarr.open(zpath_input, mode='r')
        crop_size = tuple(
            min(crop_size[i], img_size[i]) for i in range(self.n_dims))
        self.img_size = tuple(img_size)
        self.crop_size = crop_size
        self.rand_crop_ranges = [(
            min(
                img_size[i],
                round(crop_size[i] * (1. - self.scale_factors[i]))
            ),
            min(
                img_size[i] + 1,
                int(crop_size[i] * (1. + self.scale_factors[i])) + 1
            )
        ) for i in range(self.n_dims)]
        self.n_crops = n_crops
        self.is_ae = is_ae

        if is_ae:
            self.is_eval = False
            self.is_livemode = False
        else:
            # Label is not used for autoencoder
            self.za_label = zarr.open(zpath_seg_label, mode='r')
            self.zpath_seg_label = zpath_seg_label
            self.indices = indices
            self.is_eval = is_eval
            self.is_livemode = is_livemode and RUN_ON_FLASK
            if self.is_livemode:
                assert redis_client is not None
                redis_client.set(REDIS_KEY_NCROPS, str(n_crops))
            if (adaptive_length and (length is not None) and
                    (len(indices) * self.n_crops <= length)):
                length = None
        if self.is_livemode:
            self.length = int(redis_client.get(REDIS_KEY_NCROPS))
        else:
            self.length = length
        if cache_maxbytes:
            self.use_cache = True
            self.cache_dict_input = LRUCacheDict(cache_maxbytes // 2)
            self.cache_dict_label = LRUCacheDict(cache_maxbytes // 2)
        else:
            self.use_cache = False
            self.cache_dict_input = None
            self.cache_dict_label = None
        self.memmap_dir = memmap_dir

    def __len__(self):
        if self.length is not None:
            return self.length
        if self.is_ae:
            return self.n_crops
        return len(self.indices) * self.n_crops

    def _get_memmap_or_load_label(self, timepoint, img_size=None):
        if self.memmap_dir:
            key = f'{Path(self.za_label.store.path).parent.name}-t{timepoint}'
            if img_size is not None:
                key += '-' + '-'.join(map(str, img_size))
            key += '-seglabel'
            fpath = Path(self.memmap_dir) / f'{key}.dat'
            lock = FileLock(str(fpath) + '.lock')
            with lock:
                if not fpath.exists():
                    logger().info(f'creating {fpath}')
                    fpath.parent.mkdir(parents=True, exist_ok=True)
                    np.memmap(
                        fpath,
                        dtype='uint8',
                        mode='w+',
                        shape=(self.za_label.shape[1:] if img_size is None
                               else img_size)
                    )[:] = (
                        self.za_label[timepoint] if img_size is None else
                        F.interpolate(
                            torch.from_numpy(
                                self.za_label[timepoint]
                            )[None, None],
                            size=img_size,
                            mode='nearest',
                        )[0, 0].numpy()
                    )
                logger().info(f'loading from {fpath}')
                return np.memmap(
                    fpath,
                    dtype='uint8',
                    mode='c',
                    shape=(self.za_label.shape[1:] if img_size is None
                           else img_size)
                )
        return (
            self.za_label[timepoint] if img_size is None else
            F.interpolate(
                torch.from_numpy(self.za_label[timepoint])[None, None],
                size=img_size,
                mode='nearest',
            )[0, 0].numpy()
        )

    def _get_label_at(self, ind, img_size=None):
        if self.use_cache:
            za_label_a = zarr.open(self.zpath_seg_label, mode='a')
            if za_label_a.attrs.get('updated', False):
                self.cache_dict_label.clear()
                za_label_a.attrs['updated'] = False
            key = f'{self.za_label.store.path}-t{ind}'
            if img_size is not None:
                key += '-' + '-'.join(map(str, img_size))
            key += '-seglabel'
            cache = self.cache_dict_label.get(key)
            if cache is None:
                label = self._get_memmap_or_load_label(ind, img_size)
                assert 0 < label.max(), (
                    'positive weight should exist in the label'
                )
                cache = self.cache_dict_label.get(key, label)
            return cache
        label = self._get_memmap_or_load_label(ind, img_size)
        assert 0 < label.max(), (
            'positive weight should exist in the label'
        )
        return label

    def __getitem__(self, index):
        """
        Input shape: ((D,) H, W)
        Label shape: ((D,) H, W)
        Label values: 0: unlabeled, 1: BG (LW), 2: Outer (LW), 3: Inner (LW)
                                    4: BG (HW), 5: Outer (HW), 6: Inner (HW)
        """
        if (redis_client is not None and
                get_state() == TrainState.IDLE.value):
            raise KeyboardInterrupt
        if self.is_ae:
            i_frame = randrange(self.za_input.shape[0])
        else:
            if self.is_livemode:
                while True:
                    v = redis_client.get(REDIS_KEY_TIMEPOINT)
                    if v is not None:
                        i_frame = int(v)
                        img_label = self._get_label_at(i_frame, self.img_size)
                        break
                    if (get_state() == TrainState.IDLE.value):
                        raise KeyboardInterrupt
                    if self.length != int(redis_client.get(REDIS_KEY_NCROPS)):
                        return ((torch.tensor(-200.), self.keep_axials),
                                torch.tensor(-200))
            else:
                if self.length is not None:
                    i_frame = np.random.choice(self.indices)
                else:
                    i_frame = self.indices[index // self.n_crops]
                img_label = self._get_label_at(i_frame, self.img_size)
        img_input = get_input_at(self.za_input, i_frame, self.cache_dict_input,
                                 self.memmap_dir, img_size=self.img_size)
        self.i_frame = i_frame
        if self.za_input.shape[1:] != self.img_size:
            self.resize_factor = [self.img_size[d] / self.za_input.shape[1+d]
                                  for d in range(img_input.ndim)]
        else:
            self.resize_factor = [1, ] * img_input.ndim

        return super()._generate_item(img_input, img_label, self.crop_size)


class SegmentationDatasetNumpy(SegmentationDatasetBase):
    def __init__(self, images, labels, crop_size=(96, 96),
                 keep_axials=(True,) * 4, scales=None, scale_factor_base=0,
                 rotation_angle=None, contrast=0.5, is_eval=False):
        """Generate dataset for segmentation.

        Args:
            images(list of ndarray): input images.
            labels(list of ndarray): label images corresponding to input images.
            crop_size(array-like of length ndim): crop size to generate dataset.
            keep_axials(array-like of length 4): this value is used to calculate
                how many times down/up sampling are performed in z direction.
                Ignored for 2D data.
            scales(array-like of length ndim): a list of pixel/voxel size in
                physical unit (e.g. 0.5 μm/px). This is used to calculate scale
                factors for augmentation.
            scale_factor_base(float): a base scale factor for augmentation.
            rotation_angle(float): rotation angle for augmentation in degree.
            contrast(float): contrast factor for augmentation.
            is_eval(boolean): True if the dataset is for evaluation, where no
                augmentation is performed.
        """
        if len(images) != len(labels):
            raise ValueError(
                'len(images): {} and len(labels): {} should be the same'
                .format(len(images), len(labels))
            )
        for img, lbl in zip(images, labels):
            if img.shape != lbl.shape:
                raise ValueError(
                    'img.shape: {} and lbl.shape: {} should be the same'
                    .format(img.shape, lbl.shape)
                )
        super().__init__(crop_size, keep_axials, scales, scale_factor_base,
                         rotation_angle, contrast, is_eval)
        self.resize_factor = self.resize_factor = [1, ] * self.n_dims
        self.is_ae = False
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        """
        Input shape: ((D,) H, W)
        Label shape: ((D,) H, W)
        Label values: 0: unlabeled, 1: BG (LW), 2: Outer (LW), 3: Inner (LW)
                                    4: BG (HW), 5: Outer (HW), 6: Inner (HW)
        """
        img_input = self.images[index]
        img_label = self.labels[index]
        crop_size = tuple(
            min(self.crop_size[i], img_input.shape[i])
            for i in range(self.n_dims)
        )

        return super()._generate_item(img_input, img_label, crop_size)


class AutoencoderDatasetZarr(SegmentationDatasetZarr):
    def __init__(self, zpath_input, img_size, crop_size, n_crops,
                 keep_axials=(True,) * 4, scales=None, scale_factor_base=0.2,
                 cache_maxbytes=None, memmap_dir=None):
        """Generate dataset for self-supervised training for segmentation.

        Args:
            zpath_input(String): a path to .zarr file for input data.
            img_size(array-like of length ndim): input image size.
            crop_size(array-like of length ndim): crop size to generate dataset.
            n_crops(int): number of crops per timepoint.
            keep_axials(array-like of length 4): this value is used to calculate
                how many times down/up sampling are performed in z direction.
                Ignored for 2D data.
            scales(array-like of length ndim): a list of pixel/voxel size in
                physical unit (e.g. 0.5 μm/px). This is used to calculate scale
                factors for augmentation.
            cache_maxbytes (int): size of the memory capacity for cache in byte.
            memmap_dir (str): path to a directory for storing memmap files.
        """
        super().__init__(zpath_input,
                         None,
                         None,
                         img_size,
                         crop_size,
                         n_crops,
                         keep_axials=keep_axials,
                         scales=scales,
                         scale_factor_base=scale_factor_base,
                         is_ae=True,
                         cache_maxbytes=cache_maxbytes,
                         memmap_dir=memmap_dir)


class FlowDatasetZarr(du.Dataset):
    def __init__(self, zpath_input, zpath_flow_label, indices, img_size,
                 crop_size, n_crops, keep_axials=(True,) * 4, scales=None,
                 scale_factor_base=0.2, rotation_angle=None, is_eval=False,
                 length=None, adaptive_length=False, cache_maxbytes=None,
                 memmap_dir=None):
        """Generate dataset for flow estimation.

        Args:
            zpath_input(String): a path to .zarr file for input data.
            zpath_seg_label(String): a path to .zarr file for label data.
            indices(list): a list of timepoints to be used.
            img_size(array-like of length ndim): input image size.
            crop_size(array-like of length ndim): crop size to generate dataset.
            n_crops(int): number of crops per timepoint.
            keep_axials(array-like of length 4): this value is used to calculate
                how many times down/up sampling are performed in z direction.
                Ignored for 2D data.
            scales(array-like of length ndim): a list of pixel/voxel size in
                physical unit (e.g. 0.5 μm/px). This is used to calculate scale
                factors for augmentation.
            scale_factor_base(float): a base scale factor for augmentation.
            is_ae(boolean): True if called from a prior training.
            rotation_angle(float): rotation angle for augmentation in degree.
            length(int): a lenght of this dataset. If None, it is automatically
                calculated by the length of indices and n_crops.
            adaptive_length(boolean): True if the length of the dataset is
                adjusted adaptively. For example, given that the length is 10
                and the len(indices) * self.n_crops is 6, the length becomes 6
                if adaptive_length is true, while it remains 10 if false.
            cache_maxbytes (int): size of the memory capacity for cache in byte.
            memmap_dir (str): path to a directory for storing memmap files.
        """
        if len(img_size) != len(crop_size):
            raise ValueError(
                'img_size: {} and crop_size: {} should have the same length'
                .format(img_size, crop_size)
            )
        if scale_factor_base < 0 or 1 <= scale_factor_base:
            raise ValueError(
                'scale_factor_base should be 0 <= scale_factor_base < 1'
            )
        self.zpath_input = zpath_input
        self.zpath_flow_label = zpath_flow_label
        self.indices = indices
        self.is_eval = is_eval
        if (adaptive_length and (length is not None) and
                (len(indices) * self.n_crops <= length)):
            length = None
        self.length = length
        crop_size = tuple(
            min(crop_size[i], img_size[i]) for i in range(len(crop_size)))
        self.img_size = tuple(img_size)
        self.crop_size = crop_size
        self.n_crops = n_crops
        self.n_dims = len(crop_size)
        if scales is None:
            scales = (1.,) * self.n_dims
        scale_factors = tuple(
            scale_factor_base * min(scales) / scales[i]
            for i in range(self.n_dims)
        )
        self.scale_factors = scale_factors
        self.za_input = zarr.open(self.zpath_input, mode='r')
        self.za_label = zarr.open(self.zpath_flow_label, mode='a')
        self.rand_crop_ranges = [(
            min(
                img_size[i],
                round(crop_size[i] * (1. - scale_factors[i]))
            ),
            min(
                img_size[i] + 1,
                int(crop_size[i] * (1. + scale_factors[i])) + 1
            )
        ) for i in range(self.n_dims)]
        self.rotation_angle = rotation_angle
        self.keep_axials = torch.tensor(keep_axials)
        if cache_maxbytes:
            self.use_cache = True
            self.cache_dict_input = LRUCacheDict(cache_maxbytes // 2)
            self.cache_dict_label = LRUCacheDict(cache_maxbytes // 2)
        else:
            self.use_cache = False
        self.memmap_dir = memmap_dir

    def __len__(self):
        if self.length is not None:
            return self.length
        return len(self.indices) * self.n_crops

    def _get_memmap_or_load_label(self, timepoint, img_size=None):
        if self.memmap_dir:
            key = f'{Path(self.za_label.store.path).parent.name}-t{timepoint}'
            if img_size is not None:
                key += '-' + '-'.join(map(str, img_size))
            key += '-flowlabel'
            fpath = Path(self.memmap_dir) / f'{key}.dat'
            lock = FileLock(str(fpath) + '.lock')
            shape = (self.n_dims + 1, ) + (
                self.za_label.shape[-self.n_dims:] if img_size is None
                else img_size)
            with lock:
                if not fpath.exists():
                    logger().info(f'creating {fpath}')
                    fpath.parent.mkdir(parents=True, exist_ok=True)
                    np.memmap(
                        fpath,
                        dtype='float32',
                        mode='w+',
                        shape=shape,
                    )[:] = (
                        self.za_label[timepoint] if img_size is None else
                        F.interpolate(
                            torch.from_numpy(
                                self.za_label[timepoint]
                            )[None],
                            size=img_size,
                            mode='nearest',
                        )[0].numpy()
                    )
                logger().info(f'loading from {fpath}')
                return np.memmap(
                    fpath,
                    dtype='float32',
                    mode='c',
                    shape=shape,
                )
        return (
            self.za_label[timepoint] if img_size is None else
            F.interpolate(
                torch.from_numpy(self.za_label[timepoint])[None],
                size=img_size,
                mode='nearest',
            )[0].numpy()
        )

    def _get_label_at(self, ind, img_size=None):
        if self.use_cache:
            za_label_a = zarr.open(self.zpath_flow_label, mode='a')
            if za_label_a.attrs.get('updated', False):
                self.cache_dict_label.clear()
                za_label_a.attrs['updated'] = False
            key = f'{self.za_label.store.path}-t{ind}'
            if img_size is not None:
                key += '-' + '-'.join(map(str, img_size))
            key += '-flowlabel'
            cache = self.cache_dict_label.get(key)
            if cache is None:
                label = self._get_memmap_or_load_label(ind, img_size)
                assert 0 < label[-1].max(), (
                    'positive weight should exist in the label'
                )
                cache = self.cache_dict_label.get(key, label)
            return cache
        label = self._get_memmap_or_load_label(ind, img_size)
        assert 0 < label[-1].max(), (
            'positive weight should exist in the label'
        )
        return label

    @profile
    def __getitem__(self, index):
        """
        Input shape: (2, (D,) H, W)
        Label shape: (ndim+1, (D,) H, W)
        Label channels: (flow_x, flow_y, flow_z, mask)
        """
        if (redis_client is not None and
                get_state() == TrainState.IDLE.value):
            raise KeyboardInterrupt
        if self.length is not None:
            i_frame = np.random.choice(self.indices)
        else:
            i_frame = self.indices[index // self.n_crops]
        img_input = get_inputs_at(self.za_input, i_frame,
                                  cache_dict=self.cache_dict_input,
                                  memmap_dir=self.memmap_dir,
                                  img_size=self.img_size)
        if self.za_input.shape[1:] != self.img_size:
            resize_factor = [self.img_size[d] / self.za_input.shape[1+d]
                             for d in range(self.n_dims)]
        else:
            resize_factor = [1, ] * self.n_dims
        img_label = self._get_label_at(i_frame, img_size=self.img_size)
        if self.is_eval:
            tensor_input = torch.from_numpy(img_input)
            tensor_label = torch.from_numpy(img_label)
            tensor_target = torch.cat((tensor_label, tensor_input), )
            return (tensor_input, self.keep_axials), tensor_target

        while True:
            if 0 < sum(self.scale_factors):
                item_crop_size = [
                    randrange(
                        min(
                            img_input.shape[i + 1],
                            round(self.crop_size[i] *
                                  (1. - self.scale_factors[i]))
                        ),
                        min(
                            img_input.shape[i + 1] + 1,
                            int(self.crop_size[i] *
                                (1. + self.scale_factors[i])) + 1
                        )
                    ) for i in range(self.n_dims)
                ]
            else:
                item_crop_size = self.crop_size
            if self.rotation_angle is not None and 0 < self.rotation_angle:
                # rotate image
                theta = randint(-self.rotation_angle, self.rotation_angle)
                cos_theta = math.cos(math.radians(theta))
                sin_theta = math.sin(math.radians(theta))
                for i in (-2, -1):
                    item_crop_size[i] *= (abs(cos_theta) + abs(sin_theta))
                    item_crop_size[i] = math.ceil(item_crop_size[i])
                item_crop_size = [min(img_input.shape[i], item_crop_size[i])
                                  for i in range(self.n_dims)]
            za_label_a = zarr.open(self.zpath_flow_label, mode='a')
            index_pool = za_label_a.attrs.get(f'label.indices.{i_frame}')
            if index_pool is None:
                index_pool = np.argwhere(0 < img_label[-1])
                za_label_a.attrs[f'label.indices.{i_frame}'] = tuple(
                    map(tuple, index_pool.tolist())
                )
            base_index = index_pool[randrange(len(index_pool))]
            origins = [
                randint(
                    max(0,
                        (int(base_index[i] * resize_factor[i]) -
                         (item_crop_size[i] - 1))),
                    min((img_input.shape[1+i] - item_crop_size[i]),
                        int(base_index[i] * resize_factor[i]))
                )
                for i in range(self.n_dims)
            ]
            slices = (slice(None),) + tuple(
                slice(origins[i], origins[i] + item_crop_size[i])
                for i in range(self.n_dims)
            )
            sliced_label = img_label[slices].copy()
            # assert 0 < sliced_label[-1].max()
            sliced_input = img_input[slices].copy()

            # scale labels by resize factor
            if 0 < sum(self.scale_factors):
                sliced_label[0] *= self.crop_size[-1] / item_crop_size[-1]  # X
                sliced_label[1] *= self.crop_size[-2] / item_crop_size[-2]  # Y
                if self.n_dims == 3:
                    sliced_label[2] *= self.crop_size[-3] / \
                        item_crop_size[-3]  # Z

            if self.rotation_angle is not None and 0 < self.rotation_angle:
                if self.n_dims == 3:
                    sliced_input = np.array([
                        [rotate(
                            sliced_input[c, z],
                            theta,
                            resize=True,
                            preserve_range=True,
                            order=1,  # 1: Bi-linear (default)
                        ) for z in range(sliced_input.shape[1])
                        ] for c in range(sliced_input.shape[0])
                    ])
                else:
                    sliced_input = np.array([
                        rotate(
                            sliced_input[c],
                            theta,
                            resize=True,
                            preserve_range=True,
                            order=1,  # 1: Bi-linear (default)
                        ) for c in range(sliced_input.shape[0])
                    ])
                h_crop, w_crop = self.crop_size[-2:]
                h_rotate, w_rotate = sliced_input.shape[-2:]
                r_origin = max(0, (h_rotate - h_crop) // 2)
                c_origin = max(0, (w_rotate - w_crop) // 2)
                sliced_input = sliced_input[...,
                                            r_origin:r_origin+h_crop,
                                            c_origin:c_origin+w_crop]
                # rotate label
                if self.n_dims == 3:
                    sliced_label = np.array([
                        [rotate(
                            sliced_label[c, z],
                            theta,
                            resize=True,
                            preserve_range=True,
                            order=0,  # 0: Nearest-neighbor
                        ) for z in range(sliced_label.shape[1])
                        ] for c in range(sliced_label.shape[0])
                    ])
                else:
                    sliced_label = np.array([
                        rotate(
                            sliced_label[c],
                            theta,
                            resize=True,
                            preserve_range=True,
                            order=0,  # 0: Nearest-neighbor
                        ) for c in range(sliced_label.shape[0])
                    ])
                sliced_label = sliced_label[...,
                                            r_origin:r_origin+h_crop,
                                            c_origin:c_origin+w_crop]
                if sliced_label[-1].max() == 0:
                    continue
                # update flow label (y axis is inversed in the image coordinate)
                cos_theta = math.cos(math.radians(theta))
                sin_theta = math.sin(math.radians(theta))
                sliced_label_x = sliced_label[0].copy()
                sliced_label_y = sliced_label[1].copy() * -1
                sliced_label[0] = (cos_theta * sliced_label_x -
                                   sin_theta * sliced_label_y)
                sliced_label[1] = (sin_theta * sliced_label_x +
                                   cos_theta * sliced_label_y) * -1
            if 0 < sliced_label[-1].max():
                break

        tensor_input = torch.from_numpy(sliced_input)
        tensor_label = torch.from_numpy(sliced_label)
        if tensor_input.shape[1:] != self.crop_size:
            interpolate_mode = 'trilinear' if self.n_dims == 3 else 'bilinear'
            tensor_input = F.interpolate(tensor_input[None],
                                         self.crop_size,
                                         mode=interpolate_mode,
                                         align_corners=True)[0]
            tensor_label = F.interpolate(tensor_label[None].float(),
                                         self.crop_size,
                                         mode='nearest')[0]
        is_flip = True
        if is_flip:
            flip_dims = [-(1 + i) for i, v in
                         enumerate(torch.rand(self.n_dims)) if v < 0.5]
            tensor_input = torch.flip(tensor_input, flip_dims)
            tensor_label = torch.flip(tensor_label, flip_dims)
            for flip_dim in flip_dims:
                tensor_label[-1 - flip_dim] *= -1
        # Channel order: (flow_x, flow_y, flow_z, mask, input_t0, input_t1)
        tensor_target = torch.cat((tensor_label, tensor_input), )
        return (tensor_input, self.keep_axials), tensor_target
