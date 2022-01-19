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
from random import uniform
from random import randint
from random import randrange

import numpy as np
from skimage.transform import rotate
import torch
import torch.nn.functional as F
import torch.utils.data as du
if os.environ.get('CTC') != '1':
    import zarr

from elephant.util import normalize_zero_one
from elephant.logging import logger
from elephant.redis_util import REDIS_KEY_NCROPS
from elephant.redis_util import REDIS_KEY_STATE
from elephant.redis_util import REDIS_KEY_TIMEPOINT
from elephant.redis_util import TrainState


class SegmentationDatasetZarr(du.Dataset):
    def __init__(self, zpath_input, zpath_seg_label, indices, img_size,
                 crop_size, n_crops, keep_axials=(True,) * 4, scales=None,
                 is_livemode=False, redis_client=None, scale_factor_base=0.2,
                 is_ae=False, rotation_angle=None, contrast=0.5, is_eval=False,
                 length=None, adaptive_length=False):
        if len(img_size) != len(crop_size):
            raise ValueError(
                'img_size: {} and crop_size: {} should have the same length'
                .format(img_size, crop_size)
            )
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
        self.zpath_input = zpath_input
        crop_size = tuple(
            min(crop_size[i], img_size[i]) for i in range(self.n_dims))
        self.img_size = img_size
        self.crop_size = crop_size
        self.scale_factors = scale_factors
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
        self.n_crops = n_crops
        self.is_ae = is_ae
        # Not used for autoencoder
        if not is_ae:
            self.zpath_seg_label = zpath_seg_label
            self.indices = indices
            self.is_livemode = is_livemode
            if is_livemode:
                assert redis_client is not None
                redis_client.set(REDIS_KEY_NCROPS, str(n_crops))
                self.redis_c = redis_client
            if (adaptive_length and (length is not None) and
                    (len(indices) <= length)):
                length = None
        self.rotation_angle = rotation_angle
        self.contrast = contrast
        self.is_eval = is_eval
        if self.is_livemode:
            self.length = int(self.redis_c.get(REDIS_KEY_NCROPS))
        else:
            self.length = length
        self.keep_axials = torch.tensor(keep_axials)

    def __len__(self):
        if self.length is not None:
            return self.length
        if self.is_ae:
            return self.n_crops
        return len(self.indices) * self.n_crops

    def _is_valid(self, input):
        # prevent from over-fitting to outer labels
        # If the input contains outer, it should also contains inner.
        if input.max() < 0:
            return False
        if np.isin(np.unique(input), [1, 4]).any():
            return np.isin(np.unique(input), [2, 5]).any()
        return True

    def __getitem__(self, index):
        za_input = zarr.open(self.zpath_input, mode='r')
        if self.is_ae:
            i_frame = randrange(za_input.shape[0])
        else:
            # 0: unlabeled, 1: BG (LW), 2: Outer (LW), 3: Inner (LW)
            #               4: BG (HW), 5: Outer (HW), 6: Inner (HW)
            za_label = zarr.open(self.zpath_seg_label, mode='r')
            if self.is_livemode:
                while True:
                    v = self.redis_c.get(REDIS_KEY_TIMEPOINT)
                    if v is not None:
                        i_frame = int(v)
                        logger().debug('receive', i_frame)
                        if 0 < za_label[i_frame].max():
                            break
                    if (int(self.redis_c.get(REDIS_KEY_STATE)) ==
                            TrainState.IDLE.value):
                        return ((torch.tensor(-100.), self.keep_axials),
                                torch.tensor(-100))
                    if self.length != int(self.redis_c.get(REDIS_KEY_NCROPS)):
                        return ((torch.tensor(-200.), self.keep_axials),
                                torch.tensor(-200))
            elif self.length is not None:
                i_frame = np.random.choice(self.indices)
            else:
                i_frame = self.indices[index // self.n_crops]

            img_label = za_label[i_frame].astype('int64')
            # 0: unlabeled, 1: BG (LW), 2: Outer (LW), 3: Inner (LW)
            #               4: BG (HW), 5: Outer (HW), 6: Inner (HW)
            assert 0 < img_label.max(), (
                'positive weight should exist in the label'
            )
        img_input = normalize_zero_one(za_input[i_frame].astype('float32'))
        if self.is_eval:
            tensor_input = torch.from_numpy(img_input[None])
            tensor_label = torch.from_numpy(img_label - 1).long()
            return (tensor_input, self.keep_axials), tensor_label
        if not self.is_ae and self.contrast:
            fg_index = np.isin(img_label, (1, 2, 4, 5))
            bg_index = np.isin(img_label, (0, 3))
            if fg_index.any() and bg_index.any():
                fg_mean = img_input[fg_index].mean()
                bg_mean = img_input[bg_index].mean()
                cr_factor = (((fg_mean - bg_mean) * uniform(0.5, 1) + bg_mean)
                             / fg_mean)
                img_input[fg_index] *= cr_factor
        if self.rotation_angle is not None and 0 < self.rotation_angle:
            # rotate image
            theta = randint(-self.rotation_angle, self.rotation_angle)
            if self.n_dims == 3:
                img_input = np.array([
                    rotate(
                        img_input[z],
                        theta,
                        resize=True,
                        preserve_range=True,
                        order=1,  # 1: Bi-linear (default)
                    ) for z in range(img_input.shape[0])
                ])
            else:
                img_input = rotate(img_input,
                                   theta,
                                   resize=True,
                                   preserve_range=True,
                                   order=1,  # 1: Bi-linear (default)
                                   )

            # rotate label
            if self.n_dims == 3:
                img_label = np.array([
                    rotate(
                        img_label[z],
                        theta,
                        resize=True,
                        preserve_range=True,
                        order=0,  # 0: Nearest-neighbor
                    ) for z in range(img_label.shape[0])
                ])
            else:
                img_label = rotate(img_label,
                                   theta,
                                   resize=True,
                                   preserve_range=True,
                                   order=0,  # 0: Nearest-neighbor
                                   )
        if 0 < sum(self.scale_factors):
            item_crop_size = [
                randrange(
                    min(
                        img_input.shape[i],
                        round(self.crop_size[i] * (1. - self.scale_factors[i]))
                    ),
                    min(
                        img_input.shape[i] + 1,
                        int(self.crop_size[i] *
                            (1. + self.scale_factors[i])) + 1
                    )
                ) for i in range(self.n_dims)
            ]
        else:
            item_crop_size = self.crop_size
        if not self.is_ae:
            img_label -= 1
            # -1: unlabeled, 0: BG (LW), 1: Outer (LW), 2: Inner (LW)
            #                3: BG (HW), 4: Outer (HW), 5: Inner (HW)
            index_pool = np.argwhere(-1 < img_label)
        while True:
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
                            base_index[i] - (item_crop_size[i] - 1)),
                        min(img_input.shape[i] - item_crop_size[i],
                            base_index[i])
                    )
                    for i in range(self.n_dims)
                ]
            slices = tuple(
                slice(origins[i], origins[i] + item_crop_size[i])
                for i in range(self.n_dims)
            )
            if self.is_ae:
                break
            # First screening
            if self._is_valid(img_label[slices]):
                tensor_label = torch.from_numpy(img_label[slices])
                if 0 < sum(self.scale_factors):
                    tensor_label = F.interpolate(
                        tensor_label[None, None].float(),
                        self.crop_size,
                        mode='nearest'
                    ).long()
                else:
                    tensor_label = tensor_label[None, None].long()
                # Second screening
                if self._is_valid(tensor_label.numpy()):
                    break
        tensor_input = torch.from_numpy(img_input[slices])
        if 0 < sum(self.scale_factors):
            interpolate_mode = 'trilinear' if self.n_dims == 3 else 'bilinear'
            tensor_input = F.interpolate(tensor_input[None, None],
                                         self.crop_size,
                                         mode=interpolate_mode,
                                         align_corners=True)
            tensor_input = tensor_input.view((1,) + self.crop_size)
        else:
            tensor_input = tensor_input[None]
        if self.is_ae:
            return (tensor_input, self.keep_axials), tensor_input
        tensor_label = tensor_label.view(self.crop_size)
        flip_dims = [-(1 + i)
                     for i, v in enumerate(torch.rand(self.n_dims)) if v < 0.5]
        tensor_input.data = torch.flip(tensor_input, flip_dims)
        tensor_label.data = torch.flip(tensor_label, flip_dims)
        return (tensor_input, self.keep_axials), tensor_label


class AutoencoderDatasetZarr(SegmentationDatasetZarr):
    def __init__(self, zpath_input, img_size, crop_size, n_crops,
                 keep_axials=(True,) * 4, scales=None, scale_factor_base=0.2):
        super().__init__(zpath_input,
                         None,
                         None,
                         img_size,
                         crop_size,
                         n_crops,
                         keep_axials=keep_axials,
                         scales=scales,
                         scale_factor_base=scale_factor_base,
                         is_ae=True)


class FlowDatasetZarr(du.Dataset):
    def __init__(self, zpath_input, zpath_flow_label, indices, img_size,
                 crop_size, n_crops, keep_axials=(True,) * 4, scales=None,
                 scale_factor_base=0.2, rotation_angle=None, is_eval=False,
                 length=None, adaptive_length=False):
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
                (len(indices) <= length)):
            length = None
        self.length = length
        crop_size = tuple(
            min(crop_size[i], img_size[i]) for i in range(len(crop_size)))
        self.img_size = img_size
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

    def __len__(self):
        if self.length is not None:
            return self.length
        return len(self.indices) * self.n_crops

    def __getitem__(self, index):
        if self.length is not None:
            i_frame = np.random.choice(self.indices)
        else:
            i_frame = self.indices[index // self.n_crops]
        za_input = zarr.open(self.zpath_input, mode='r')
        img_input = np.array([
            normalize_zero_one(za_input[i].astype('float32'))
            for i in range(i_frame, i_frame + 2)
        ])
        za_label = zarr.open(self.zpath_flow_label, mode='r')
        img_label = za_label[i_frame]
        assert 0 < img_label[-1].max(), (
            'positive weight should exist in the label')
        if self.is_eval:
            tensor_input = torch.from_numpy(img_input)
            tensor_label = torch.from_numpy(img_label)
            tensor_target = torch.cat((tensor_label, tensor_input), )
            return (tensor_input, self.keep_axials), tensor_target
        if self.rotation_angle is not None and 0 < self.rotation_angle:
            # rotate image
            theta = randint(-self.rotation_angle, self.rotation_angle)
            if self.n_dims == 3:
                img_input = np.array([
                    [rotate(
                        img_input[c, z],
                        theta,
                        resize=True,
                        preserve_range=True,
                        order=1,  # 1: Bi-linear (default)
                    ) for z in range(img_input.shape[1])
                    ] for c in range(img_input.shape[0])
                ])
            else:
                img_input = np.array([
                    rotate(
                        img_input[c],
                        theta,
                        resize=True,
                        preserve_range=True,
                        order=1,  # 1: Bi-linear (default)
                    ) for c in range(img_input.shape[0])
                ])
            # rotate label
            if self.n_dims == 3:
                img_label = np.array([
                    [rotate(
                        img_label[c, z],
                        theta,
                        resize=True,
                        preserve_range=True,
                        order=0,  # 0: Nearest-neighbor
                    ) for z in range(img_label.shape[1])
                    ] for c in range(img_label.shape[0])
                ])
            else:
                img_label = np.array([
                    rotate(
                        img_label[c],
                        theta,
                        resize=True,
                        preserve_range=True,
                        order=0,  # 0: Nearest-neighbor
                    ) for c in range(img_label.shape[0])
                ])
            # update flow label (y axis is inversed in the image coordinate)
            cos_theta = math.cos(math.radians(theta))
            sin_theta = math.sin(math.radians(theta))
            img_label_x = img_label[0].copy()
            img_label_y = img_label[1].copy() * -1
            img_label[0] = cos_theta * img_label_x - sin_theta * img_label_y
            img_label[1] = (sin_theta * img_label_x +
                            cos_theta * img_label_y) * -1

        if 0 < sum(self.scale_factors):
            item_crop_size = [
                randrange(
                    min(
                        img_input.shape[i + 1],
                        round(self.crop_size[i] * (1. - self.scale_factors[i]))
                    ),
                    min(
                        img_input.shape[i + 1] + 1,
                        int(self.crop_size[i] *
                            (1. + self.scale_factors[i])) + 1
                    )
                ) for i in range(self.n_dims)
            ]
            # scale labels by resize factor
            img_label[0] *= self.crop_size[-1] / item_crop_size[-1]  # X
            img_label[1] *= self.crop_size[-2] / item_crop_size[-2]  # Y
            if self.n_dims == 3:
                img_label[2] *= self.crop_size[-3] / item_crop_size[-3]  # Z
        else:
            item_crop_size = self.crop_size
        index_pool = np.argwhere(0 < img_label[-1])
        while True:
            base_index = index_pool[randrange(len(index_pool))]
            origins = [
                randint(
                    max(0,
                        base_index[i] - (item_crop_size[i] - 1)),
                    min(img_input.shape[i + 1] - item_crop_size[i],
                        base_index[i])
                )
                for i in range(self.n_dims)
            ]
            slices = (slice(None),) + tuple(
                slice(origins[i], origins[i] + item_crop_size[i])
                for i in range(self.n_dims)
            )
            # First screening
            if img_label[slices][-1].max() != 0:
                tensor_label = torch.from_numpy(img_label[slices])
                if 0 < sum(self.scale_factors):
                    tensor_label = F.interpolate(tensor_label[None].float(),
                                                 self.crop_size,
                                                 mode='nearest')
                    tensor_label = tensor_label.view(
                        (self.n_dims + 1,) + self.crop_size)
                # Second screening
                if tensor_label[-1].max() != 0:
                    break
        tensor_input = torch.from_numpy(img_input[slices])
        if 0 < sum(self.scale_factors):
            interpolate_mode = 'trilinear' if self.n_dims == 3 else 'bilinear'
            tensor_input = F.interpolate(tensor_input[None],
                                         self.crop_size,
                                         mode=interpolate_mode,
                                         align_corners=True)
            tensor_input = tensor_input.view((2,) + self.crop_size)
        # Channel order: (flow_x, flow_y, flow_z, mask, input_t0, input_t1)
        tensor_target = torch.cat((tensor_label, tensor_input), )
        return (tensor_input, self.keep_axials), tensor_target
