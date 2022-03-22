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
"""Convert configs in json to python objects."""

import json
import os
from collections import OrderedDict
from pathlib import Path

import torch

from elephant.util import get_device

DATASETS_DIR = '/workspace/datasets'
MODELS_DIR = '/workspace/models'
LOGS_DIR = '/workspace/logs'
MEMMAPS_DIR = '/workspace/memmaps'

ZARR_INPUT = 'imgs.zarr'
ZARR_SEG_OUTPUT = 'seg_outputs.zarr'
ZARR_SEG_LABELS = 'seg_labels.zarr'
ZARR_SEG_LABELS_VIS = 'seg_labels_vis.zarr'
ZARR_FLOW = 'flow_outputs.zarr'
ZARR_FLOW_LABELS = 'flow_labels.zarr'
ZARR_FLOW_HASHES = 'flow_hashes.zarr'

DEFAULT_MAX_DISPLACEMENT = 80
DEFAULT_CROP_SIZE = (384, 384, 16)  # X, Y, Z
DEFAULT_SEG_CLASS_WEIGHTS = (1., 10., 10.)  # BG, Border, FG
DEFAULT_FLOW_DIM_WEIGHTS = (0.1, 0.1, 0.8)  # X, Y, Z


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


class BaseConfig():
    def __init__(self, config):
        self.dataset_name = config.get('dataset_name')
        self.timepoint = config.get('timepoint')
        if config.get('model_name') is not None:
            self.model_path = os.path.join(
                MODELS_DIR, config.get('model_name'))
        self.device = config.get('device', get_device())
        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
        self.is_mixed_precision = config.get('is_mixed_precision', True)
        self.debug = config.get('debug', False)
        self.output_prediction = config.get('output_prediction', False)
        self.is_3d = config.get('is_3d', True)
        self.use_2d = config.get('use_2d', False)
        self.batch_size = config.get('batch_size', 1)
        self.patch_size = config.get('patch')
        self.log_interval = config.get('log_interval', 10)
        self.cache_maxbytes = config.get('cache_maxbytes')
        if self.patch_size is not None:
            self.patch_size = self.patch_size[::-1]
        if config.get('scales') is not None:
            self.scales = config.get('scales')[::-1]
        else:
            self.scales = None
        if config.get('input_size') is not None:
            self.input_size = tuple(config.get('input_size')[::-1])
        else:
            self.input_size = None
        # U-Net has 4 downsamplings
        n_keep_axials = min(4, config.get('n_keep_axials', 4))
        self.keep_axials = tuple(True if i < n_keep_axials else False
                                 for i in range(4))
        self.crop_size = config.get('crop_size', DEFAULT_CROP_SIZE)[::-1]
        if not self.is_3d:
            if self.patch_size is not None:
                self.patch_size = self.patch_size[-2:]
            if self.scales is not None:
                self.scales = self.scales[-2:]
            self.crop_size = self.crop_size[-2:]
            if self.input_size is not None:
                self.input_size = self.input_size[-2:]
        if self.dataset_name is not None:
            self.zpath_input = os.path.join(DATASETS_DIR,
                                            self.dataset_name,
                                            ZARR_INPUT)
        else:
            self.zpath_input = None
        if config.get('use_memmap'):
            self.memmap_dir = MEMMAPS_DIR
            Path(self.memmap_dir).mkdir(parents=True, exist_ok=True)
        else:
            self.memmap_dir = None

    def is_cpu(self):
        return str(self.device) == 'cpu'

    def __str__(self):
        result = []
        for k, v in sorted(vars(self).items()):
            result.append(f'{k}: {v}')
        return '\n'.join(result)


class ResetConfig(BaseConfig):
    def __init__(self, config):
        super().__init__(config)
        self.url = config.get('url', None)
        self.n_models = config.get('nmodels', 1)
        self.n_crops = config.get('n_crops', 5)


class ExportConfig(BaseConfig):
    def __init__(self, config):
        super().__init__(config)
        self.savedir = config.get('savedir')
        self.shape = config.get('shape')
        t_start = config.get('t_start')
        t_end = config.get('t_end')
        if t_end < t_start:
            raise ValueError(
                f't_end: {t_end} should be greater than t_start: {t_start}'
            )
        self.t_start = t_start
        self.t_end = t_end


class SegmentationEvalConfig(BaseConfig):
    def __init__(self, config):
        super().__init__(config)
        self.use_median = config.get('use_median', False)
        self.is_pad = config.get('is_pad', False)
        self.c_ratio = config.get('c_ratio', 0.4)
        self.p_thresh = config.get('p_thresh', 0.5)
        self.r_min = config.get('r_min', 0)
        self.r_max = config.get('r_max', 1e6)
        crop_box = config.get('crop_box')
        if crop_box is not None:
            crop_box = crop_box[:3][::-1] + crop_box[3:][::-1]
        self.crop_box = crop_box
        if self.dataset_name is not None:
            self.zpath_seg_output = os.path.join(DATASETS_DIR,
                                                 self.dataset_name,
                                                 ZARR_SEG_OUTPUT)


class SegmentationEvalConfigTiff(SegmentationEvalConfig):
    def __init__(self, config):
        super().__init__(config)
        self.zpath_input = None
        self.zpath_seg_output = None
        self.tiff_input = config.get('tiffinput')
        self.model_path = config.get('seg_model')


class SegmentationTrainConfig(SegmentationEvalConfig):
    def __init__(self, config):
        super().__init__(config)
        self.lr = config.get('lr')
        self.n_crops = config.get('n_crops')
        self.n_epochs = config.get('n_epochs')
        self.epoch_start = config.get('epoch_start', 0)
        self.is_livemode = config.get('is_livemode', False)
        self.auto_bg_thresh = config.get('auto_bg_thresh')
        self.scale_factor_base = config.get('aug_scale_factor_base')
        self.rotation_angle = config.get('aug_rotation_angle')
        self.contrast = config.get('aug_contrast', 0)
        if self.contrast < 0 or 1 < self.contrast:
            raise ValueError(
                f'contrast should be in [0, 1] but got {self.contrast}')
        self.class_weights = config.get('class_weights',
                                        DEFAULT_SEG_CLASS_WEIGHTS)
        self.false_weight = config.get('false_weight', 10)
        if config.get('log_dir') is not None:
            self.log_dir = os.path.join(LOGS_DIR, config.get('log_dir'))
        if self.dataset_name is not None:
            self.zpath_seg_label = os.path.join(DATASETS_DIR,
                                                self.dataset_name,
                                                ZARR_SEG_LABELS)
            self.zpath_seg_label_vis = os.path.join(DATASETS_DIR,
                                                    self.dataset_name,
                                                    ZARR_SEG_LABELS_VIS)


class FlowEvalConfig(BaseConfig):
    def __init__(self, config):
        super().__init__(config)
        self.use_median = config.get('use_median', False)
        self.is_pad = config.get('is_pad', False)
        max_displacement = config.get('max_displacement',
                                      DEFAULT_MAX_DISPLACEMENT)
        self.flow_norm_factor = (
            float(max_displacement),  # X
            float(max_displacement),  # Y
            float(max_displacement / (2**config.get('n_keep_axials', 0))))  # Z
        crop_box = config.get('crop_box')
        if crop_box is not None:
            crop_box = crop_box[:3][::-1] + crop_box[3:][::-1]
        self.crop_box = crop_box
        if self.dataset_name is not None:
            self.zpath_flow = os.path.join(DATASETS_DIR,
                                           self.dataset_name,
                                           ZARR_FLOW)
            self.zpath_flow_hashes = os.path.join(DATASETS_DIR,
                                                  self.dataset_name,
                                                  ZARR_FLOW_HASHES)


class FlowEvalConfigTiff(FlowEvalConfig):
    def __init__(self, config):
        super().__init__(config)
        self.zpath_input = None
        self.zpath_flow = None
        self.zpath_flow_hashes = None
        self.tiff_input = config.get('tiffinput')
        self.model_path = config.get('flow_model')


class FlowTrainConfig(FlowEvalConfig):
    def __init__(self, config):
        super().__init__(config)
        self.lr = config.get('lr')
        self.n_crops = config.get('n_crops')
        self.n_epochs = config.get('n_epochs')
        self.epoch_start = config.get('epoch_start', 0)
        self.dim_weights = config.get('dim_weights', DEFAULT_FLOW_DIM_WEIGHTS)
        self.scale_factor_base = config.get('aug_scale_factor_base')
        self.rotation_angle = config.get('aug_rotation_angle')
        if config.get('log_dir') is not None:
            self.log_dir = os.path.join(LOGS_DIR, config.get('log_dir'))
        if self.dataset_name is not None:
            self.zpath_flow_label = os.path.join(DATASETS_DIR,
                                                 self.dataset_name,
                                                 ZARR_FLOW_LABELS)
