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

import torch
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data import DataLoader
import zarr

from elephant.common import TensorBoard
from elephant.common import evaluate
from elephant.common import load_flow_models
from elephant.common import load_seg_models
from elephant.common import train
from elephant.config import FlowTrainConfig
from elephant.config import ResetConfig
from elephant.config import SegmentationTrainConfig
from elephant.datasets import FlowDatasetZarr
from elephant.datasets import SegmentationDatasetZarr
from elephant.losses import FlowLoss
from elephant.losses import SegmentationLoss


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
        for ti, t in enumerate(range(
                t_min,
                t_max + (args.command == 'seg'))):
            if eval_interval is not None and eval_interval == -1:
                train_index.append(t)
                eval_index.append(t)
            elif eval_interval is not None and (ti + 1) % eval_interval == 0:
                eval_index.append(t)
            else:
                train_index.append(t)
        if args.command == 'seg':
            config = SegmentationTrainConfig(config_dict)
            print(config)
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
            ))
        elif args.command == 'flow':
            config = FlowTrainConfig(config_dict)
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
            ))
            eval_datasets.append(FlowDatasetZarr(
                config.zpath_input,
                config.zpath_flow_label,
                eval_index,
                input_shape,
                config.crop_size,
                config.n_crops,
                keep_axials=config.keep_axials,
                scales=config.scales,
                scale_factor_base=0.0,
            ))
    train_dataset = ConcatDataset(train_datasets)
    eval_dataset = ConcatDataset(eval_datasets)
    if 0 < len(train_dataset):
        logger = TensorBoard(config.log_dir)
        if args.command == 'seg':
            weight_tensor = torch.tensor(config.class_weights)
            loss_fn = SegmentationLoss(class_weights=weight_tensor,
                                       false_weight=config.false_weight,
                                       is_3d=config.is_3d)
        elif args.command == 'flow':
            loss_fn = FlowLoss(is_3d=config.is_3d)
        loss_fn = loss_fn.to(config.device)
        optimizers = [torch.optim.Adam(
            model.parameters(), lr=config.lr) for model in models]
        train_loader = DataLoader(
            train_dataset, shuffle=True, batch_size=config.batch_size)
        eval_loader = DataLoader(
            eval_dataset, shuffle=False, batch_size=config.batch_size)
        if 0 < len(eval_loader):
            min_loss = sum([evaluate(model,
                                     config.device,
                                     eval_loader,
                                     loss_fn=loss_fn,
                                     epoch=-1,
                                     patch_size=config.patch_size)
                            for model in models]) / len(models)
            state_dicts = ([models[0].state_dict()] if len(models) == 1
                           else
                           [model.state_dict() for model in models])

        for epoch in range(config.epoch_start,
                           config.epoch_start + config.n_epochs):
            if args.command == 'flow' and epoch == 50:
                optimizers = [
                    torch.optim.Adam(model.parameters(), lr=config.lr*0.1)
                    for model in models
                ]
            loss = 0
            for model, optimizer in zip(models, optimizers):
                train(model,
                      config.device,
                      train_loader,
                      optimizer=optimizer,
                      loss_fn=loss_fn,
                      epoch=epoch,
                      log_interval=config.log_interval,
                      tb_logger=logger)
                if 0 < len(eval_loader):
                    loss += evaluate(model,
                                     config.device,
                                     eval_loader,
                                     loss_fn=loss_fn,
                                     epoch=epoch,
                                     tb_logger=logger,
                                     patch_size=config.patch_size)
            loss /= len(models)
            if 0 < len(eval_loader):
                if loss < min_loss:
                    min_loss = loss
                    state_dicts = (models[0].state_dict() if len(models) == 1
                                   else
                                   [model.state_dict() for model in models])
                    torch.save(state_dicts,
                               config.model_path.replace('.pth', '_best.pth'))
                    if len(models) == 1:
                        state_dicts = [state_dicts]
                else:
                    pass
                    # for model, state_dict in zip(models, state_dicts):
                    #     model.load_state_dict(state_dict)
            torch.save(models[0].state_dict() if len(models) == 1 else
                       [model.state_dict() for model in models],
                       config.model_path)


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
