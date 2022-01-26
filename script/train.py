#! /usr/bin/env python
# ==============================================================================
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
"""Commandline interface for training using a config file."""
import argparse
import io
import json
import os
from pathlib import Path

from tensorflow.data import TFRecordDataset
from tensorflow.core.util import event_pb2
import torch
import torch.multiprocessing as mp
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data import DataLoader
import zarr

from elephant.common import load_flow_models
from elephant.common import load_seg_models
from elephant.common import run_train
from elephant.config import FlowTrainConfig
from elephant.config import ResetConfig
from elephant.config import SegmentationTrainConfig
from elephant.datasets import FlowDatasetZarr
from elephant.datasets import SegmentationDatasetZarr
from elephant.losses import FlowLoss
from elephant.losses import SegmentationLoss

PROFILE = "ELEPHANT_PROFILE" in os.environ


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', help='seg | flow')
    parser.add_argument('config', help='config file')
    parser.add_argument('--baseconfig', help='base config file')
    args = parser.parse_args()
    if args.command not in ['seg', 'flow']:
        print('command option should be "seg" or "flow"')
        parser.print_help()
        exit(1)
    base_config_dict = dict()
    if args.baseconfig is not None:
        with io.open(args.baseconfig, 'r', encoding='utf-8') as jsonfile:
            base_config_dict.update(json.load(jsonfile))
    with io.open(args.config, 'r', encoding='utf-8') as jsonfile:
        config_data = json.load(jsonfile)
    # load or initialize models

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # prepare dataset
    train_datasets = []
    eval_datasets = []
    for i in range(len(config_data)):
        config_dict = base_config_dict.copy()
        config_dict.update(config_data[i])
        config = ResetConfig(config_dict)
        n_dims = 2 + config.is_3d  # 3 or 2
        models = load_models(config, args.command)
        za_input = zarr.open(config.zpath_input, mode='r')
        input_shape = za_input.shape[-n_dims:]
        train_index = []
        eval_index = []
        eval_interval = config_dict.get('evalinterval')
        t_min = config_dict.get('t_min', 0)
        t_max = config_dict.get('t_max', za_input.shape[0] - 1)
        train_length = config_dict.get('train_length')
        eval_length = config_dict.get('eval_length')
        adaptive_length = config_dict.get('adaptive_length', False)
        if args.command == 'seg':
            config = SegmentationTrainConfig(config_dict)
            print(config)
            za_label = zarr.open(config.zpath_seg_label, mode='r')
            for ti, t in enumerate(range(t_min, t_max + 1)):
                if 0 < za_label[t].max():
                    if eval_interval is not None and eval_interval == -1:
                        train_index.append(t)
                        eval_index.append(t)
                    elif (eval_interval is not None and
                          (ti + 1) % eval_interval == 0):
                        eval_index.append(t)
                    else:
                        train_index.append(t)
            train_datasets.append(SegmentationDatasetZarr(
                config.zpath_input,
                config.zpath_seg_label,
                train_index,
                input_shape,
                config.crop_size,
                config.n_crops,
                keep_axials=config.keep_axials,
                scales=config.scales,
                scale_factor_base=config.scale_factor_base,
                contrast=config.contrast,
                rotation_angle=config.rotation_angle,
                length=train_length,
                cache_maxbytes=config.cache_maxbytes,
            ))
            eval_datasets.append(SegmentationDatasetZarr(
                config.zpath_input,
                config.zpath_seg_label,
                eval_index,
                input_shape,
                input_shape,
                1,
                keep_axials=config.keep_axials,
                is_eval=True,
                length=eval_length,
                adaptive_length=adaptive_length,
                cache_maxbytes=config.cache_maxbytes,
            ))
        elif args.command == 'flow':
            config = FlowTrainConfig(config_dict)
            print(config)
            za_label = zarr.open(config.zpath_flow_label, mode='r')
            for ti, t in enumerate(range(t_min, t_max)):
                if 0 < za_label[t][-1].max():
                    if eval_interval is not None and eval_interval == -1:
                        train_index.append(t)
                        eval_index.append(t)
                    elif (eval_interval is not None and
                          (ti + 1) % eval_interval == 0):
                        eval_index.append(t)
                    else:
                        train_index.append(t)
            train_datasets.append(FlowDatasetZarr(
                config.zpath_input,
                config.zpath_flow_label,
                train_index,
                input_shape,
                config.crop_size,
                config.n_crops,
                keep_axials=config.keep_axials,
                scales=config.scales,
                scale_factor_base=config.scale_factor_base,
                rotation_angle=config.rotation_angle,
                length=train_length,
            ))
            eval_datasets.append(FlowDatasetZarr(
                config.zpath_input,
                config.zpath_flow_label,
                eval_index,
                input_shape,
                input_shape,
                1,
                keep_axials=config.keep_axials,
                is_eval=True,
                length=eval_length,
                adaptive_length=adaptive_length,
            ))
    train_dataset = ConcatDataset(train_datasets)
    eval_dataset = ConcatDataset(eval_datasets)
    if 0 < len(train_dataset):
        if args.command == 'seg':
            weight_tensor = torch.tensor(config.class_weights)
            loss_fn = SegmentationLoss(class_weights=weight_tensor,
                                       false_weight=config.false_weight,
                                       is_3d=config.is_3d)
        elif args.command == 'flow':
            loss_fn = FlowLoss(is_3d=config.is_3d)
        optimizers = [torch.optim.Adam(
            model.parameters(), lr=config.lr) for model in models]
        train_loader = DataLoader(
            train_dataset, shuffle=True, batch_size=config.batch_size)
        eval_loader = DataLoader(
            eval_dataset, shuffle=False, batch_size=config.batch_size)
        step_offset = 0
        for path in sorted(Path(config.log_dir).glob('event*')):
            try:
                *_, last_record = TFRecordDataset(str(path))
                last = event_pb2.Event.FromString(last_record.numpy()).step
                step_offset = max(step_offset, last+1)
            except Exception:
                pass
        if PROFILE:
            run_train(config.device, 1, models, train_loader, optimizers,
                      loss_fn, config.n_epochs, config.model_path, False,
                      config.log_interval, config.log_dir, step_offset,
                      config.epoch_start, eval_loader, config.patch_size,
                      config.is_cpu(), True)
        else:
            world_size = 2 if config.is_cpu() else torch.cuda.device_count()
            mp.spawn(run_train,
                     args=(world_size, models, train_loader, optimizers,
                           loss_fn, config.n_epochs, config.model_path, False,
                           config.log_interval, config.log_dir, step_offset,
                           config.epoch_start, eval_loader, config.patch_size,
                           config.is_cpu(), True),
                     nprocs=world_size,
                     join=True)


def load_models(config, command):
    """ load or initialize models
    Args:
        config: config object
        command: 'seg' or 'flow'
    Returns:
        models: loaded or initialized models
    """
    if command == 'seg':
        models = load_seg_models(config.model_path,
                                 config.keep_axials,
                                 config.device,
                                 is_3d=config.is_3d,
                                 n_models=config.n_models,
                                 n_crops=config.n_crops,
                                 zpath_input=config.zpath_input,
                                 crop_size=config.crop_size,
                                 scales=config.scales,
                                 url=config.url)
    elif command == 'flow':
        models = load_flow_models(config.model_path,
                                  config.device,
                                  is_3d=config.is_3d,
                                  url=config.url)
    return models


if __name__ == '__main__':
    main()
