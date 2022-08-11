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
"""Implementations of operations."""

from collections import OrderedDict
import gc
from functools import partial
from functools import reduce
import multiprocessing as mp
import math
from operator import mul
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
from scipy import ndimage
from skimage.filters import gaussian
from skimage.filters import prewitt
import skimage.io
import skimage.measure
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.utils.data as du

if os.environ.get('CTC') != '1':
    import torch.utils.tensorboard as tb
    import zarr

from elephant.datasets import AutoencoderDatasetZarr
from elephant.datasets import get_input_at
from elephant.datasets import get_inputs_at
from elephant.datasets import FlowDatasetZarr
from elephant.datasets import SegmentationDatasetZarr
from elephant.datasets import DatasetPrediction
from elephant.logging import publish_mq
from elephant.logging import logger
from elephant.losses import AutoencoderLoss
from elephant.losses import FlowLoss
from elephant.losses import SegmentationLoss
from elephant.models import FlowResNet
from elephant.models import UNet
from elephant.redis_util import get_state
from elephant.redis_util import redis_client
from elephant.redis_util import REDIS_KEY_LR
from elephant.redis_util import REDIS_KEY_NCROPS
from elephant.redis_util import REDIS_KEY_TIMEPOINT
from elephant.redis_util import TrainState
from elephant.util import get_device
from elephant.util import normalize_zero_one
from elephant.util import to_fancy_index
from elephant.util.ellipse import ellipse
from elephant.util.ellipsoid import ellipsoid
from elephant.util.scaled_moments import scaled_moments_central

MODEL_URL_ROOT = (
    'https://github.com/elephant-track/elephant-server/releases/download/data/'
)

PROFILE = "ELEPHANT_PROFILE" in os.environ

if os.environ.get('ELEPHANT_PROFILE') is None:
    def profile(func):
        return func


class TensorBoard(object):
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.writer = tb.SummaryWriter(self.log_dir)

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag=tag, scalar_value=value, global_step=step)


def setup(rank, world_size, backend='gloo'):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    if not dist.is_initialized():
        dist.init_process_group(backend, rank=rank, world_size=world_size)
        logger().info(f"Running DDP on rank {rank} using backend {backend}.")


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def run_train_seg(rank_or_device, world_size, spot_indices, batch_size,
                  crop_size, class_weights, false_weight, model_path, n_epochs,
                  keep_axials, scales, lr, n_crops, is_3d, is_livemode,
                  scale_factor_base, rotation_angle, contrast, zpath_input,
                  zpath_seg_label, log_interval=100, log_dir=None,
                  step_offset=0, epoch_start=0, is_cpu=False,
                  is_mixed_precision=True, cache_maxbytes=None,
                  memmap_dir=None, input_size=None):
    models = load_seg_models(model_path,
                             keep_axials,
                             get_device(),
                             is_3d=is_3d,
                             zpath_input=zpath_input,
                             crop_size=crop_size,
                             scales=scales,
                             cache_maxbytes=cache_maxbytes)
    weight_tensor = torch.tensor(class_weights).float()
    loss_fn = SegmentationLoss(class_weights=weight_tensor,
                               false_weight=false_weight,
                               is_3d=is_3d)

    optimizers = [torch.optim.Adam(model.parameters(), lr=lr)
                  for model in models]

    n_dims = 2 + is_3d  # 3 or 2
    if input_size is None:
        input_size = zarr.open(zpath_input, mode='r').shape[-n_dims:]
    dataset = SegmentationDatasetZarr(
        zpath_input,
        zpath_seg_label,
        spot_indices,
        input_size,
        crop_size,
        n_crops,
        keep_axials,
        scales=scales,
        is_livemode=is_livemode,
        scale_factor_base=scale_factor_base,
        rotation_angle=rotation_angle,
        contrast=contrast,
        cache_maxbytes=cache_maxbytes,
        memmap_dir=memmap_dir,
    )
    loader = du.DataLoader(dataset, batch_size=batch_size,
                           shuffle=True, num_workers=0)
    run_train(rank_or_device, world_size, models, loader, optimizers, loss_fn,
              n_epochs, model_path, is_livemode, log_interval=log_interval,
              log_dir=log_dir, step_offset=step_offset, epoch_start=epoch_start,
              is_cpu=is_cpu, is_mixed_precision=is_mixed_precision)


def run_train_prior_seg(rank_or_device, world_size, crop_size, model_path,
                        n_epochs, keep_axials, scales, lr, n_crops,
                        is_3d, zpath_input, log_interval=100,
                        log_dir=None, step_offset=0, epoch_start=0,
                        is_cpu=False, is_mixed_precision=True,
                        cache_maxbytes=None, memmap_dir=None, input_size=None):
    models = load_seg_models(model_path,
                             keep_axials,
                             get_device(),
                             is_3d=is_3d,
                             zpath_input=zpath_input,
                             crop_size=crop_size,
                             scales=scales,
                             cache_maxbytes=cache_maxbytes)
    loss_fn = AutoencoderLoss()

    optimizers = [torch.optim.Adam(model.parameters(), lr=lr)
                  for model in models]

    n_dims = 2 + is_3d  # 3 or 2
    if input_size is None:
        input_size = zarr.open(zpath_input, mode='r').shape[-n_dims:]
    dataset = AutoencoderDatasetZarr(
        zpath_input,
        input_size,
        crop_size,
        n_crops,
        keep_axials=keep_axials,
        scales=scales,
        scale_factor_base=0.2,
        cache_maxbytes=cache_maxbytes,
        memmap_dir=memmap_dir,
    )
    loader = du.DataLoader(dataset, batch_size=1,
                           shuffle=True, num_workers=0)
    run_train(rank_or_device, world_size, models, loader, optimizers, loss_fn,
              n_epochs, model_path, log_interval=log_interval, log_dir=log_dir,
              step_offset=step_offset, epoch_start=epoch_start, is_cpu=is_cpu,
              is_mixed_precision=is_mixed_precision)


def run_train_flow(rank_or_device, world_size, spot_indices, batch_size,
                   crop_size, model_path, n_epochs, keep_axials, scales, lr,
                   n_crops, is_3d, scale_factor_base, rotation_angle,
                   zpath_input, zpath_flow_label, log_interval=100,
                   log_dir=None, step_offset=0, epoch_start=0, is_cpu=False,
                   is_mixed_precision=False, cache_maxbytes=None,
                   memmap_dir=None, input_size=None):
    models = load_flow_models(model_path,
                              get_device(),
                              is_3d=is_3d)
    loss_fn = FlowLoss(is_3d=is_3d)

    optimizers = [torch.optim.Adam(model.parameters(), lr=lr)
                  for model in models]

    n_dims = 2 + is_3d  # 3 or 2
    if input_size is None:
        input_size = zarr.open(zpath_input, mode='r').shape[-n_dims:]
    dataset = FlowDatasetZarr(
        zpath_input,
        zpath_flow_label,
        spot_indices,
        input_size,
        crop_size,
        n_crops,
        keep_axials=keep_axials,
        scales=scales,
        scale_factor_base=scale_factor_base,
        rotation_angle=rotation_angle,
        cache_maxbytes=cache_maxbytes,
        memmap_dir=memmap_dir,
    )
    loader = du.DataLoader(dataset, batch_size=batch_size, num_workers=0)
    run_train(rank_or_device, world_size, models, loader, optimizers, loss_fn,
              n_epochs, model_path, log_interval=log_interval, log_dir=log_dir,
              step_offset=step_offset, epoch_start=epoch_start, is_cpu=is_cpu,
              is_mixed_precision=is_mixed_precision)


def run_train(rank_or_device, world_size, models, loader, optimizers, loss_fn,
              n_epochs, model_path, is_livemode=False, log_interval=100,
              log_dir=None, step_offset=0, epoch_start=0, eval_loader=None,
              patch_size=None, is_cpu=False, is_mixed_precision=True):
    if not torch.cuda.is_available():
        is_cpu = True
        is_mixed_precision = False
    if is_cpu:
        device = torch.device('cpu')
        device_ids = None
        ddp_backend = 'gloo'
    else:
        device = rank_or_device
        device_ids = [device]
        ddp_backend = 'nccl'
    for model in models:
        model.train()
        model.to(device)
    is_ddp = True
    try:
        setup(rank_or_device, world_size, ddp_backend)
    except Exception:
        logger().info('DistributedDataParallel is not used.')
        is_ddp = False
        world_size = 1
    is_logging = True
    if is_ddp:
        models = [DDP(model, device_ids=device_ids) for model in models]
        loader = to_ddp_loader(loader,
                               world_size,
                               rank_or_device,
                               True)
        if eval_loader:
            eval_loader = to_ddp_loader(eval_loader,
                                        world_size,
                                        rank_or_device,
                                        False)
        # only rank=0 is used for logging
        is_logging = rank_or_device == 0
    try:
        loss_fn = loss_fn.to(device)

        if is_mixed_precision:
            scaler = torch.cuda.amp.GradScaler()

        if eval_loader:
            min_loss = sum([evaluate(model,
                                     device,
                                     eval_loader,
                                     loss_fn=loss_fn,
                                     patch_size=patch_size,
                                     is_ddp=is_ddp,
                                     is_logging=is_logging)
                            for model in models]) / len(models)
            if is_logging:
                state_dicts_best = [
                    OrderedDict({
                        k[7:]: v for k, v in model.state_dict().items()
                    }) if is_ddp else
                    model.state_dict() for model in models]

        epoch = epoch_start
        while is_livemode or epoch < epoch_start + n_epochs:
            if is_ddp:
                dist.barrier()
            if is_livemode:
                redis_client.delete(REDIS_KEY_TIMEPOINT)
            is_ncrops_updated = False
            for model, optimizer in zip(models, optimizers):
                if is_logging:
                    count = 0
                    loss_avg = 0
                    eval_loss = 0
                    if isinstance(loss_fn, SegmentationLoss):
                        nll_loss_avg = 0
                        center_loss_avg = 0
                        smooth_loss_avg = 0
                    elif isinstance(loss_fn, FlowLoss):
                        instance_loss_avg = 0
                        ssim_loss_avg = 0
                        smooth_loss_avg = 0
                for batch_id, ((x, keep_axials), y) in enumerate(loader):
                    if is_ddp:
                        dist.barrier()
                    if (torch.eq(x, torch.tensor(-200.)).any() and
                            torch.eq(y, torch.tensor(-200)).any()):
                        is_ncrops_updated = True
                        break
                    if redis_client is not None:
                        while (get_state() == TrainState.WAIT.value):
                            if is_logging:
                                logger().info('waiting @train')
                            time.sleep(1)
                        if (get_state() == TrainState.IDLE.value):
                            return
                    x, y = x.to(device), y.to(device)

                    # set leargning rate
                    if (redis_client is not None and
                            redis_client.get(REDIS_KEY_LR) is not None):
                        for g in optimizer.param_groups:
                            g['lr'] = float(redis_client.get(REDIS_KEY_LR))

                    optimizer.zero_grad()

                    if is_mixed_precision:
                        with torch.cuda.amp.autocast():
                            prediction = model(x, keep_axials)
                            loss = loss_fn(prediction, y)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        prediction = model(x, keep_axials)
                        loss = loss_fn(prediction, y)
                        loss.backward()
                        optimizer.step()

                    if is_ddp:
                        dist.all_reduce(loss)

                    if not is_logging:
                        continue

                    count += 1
                    loss_avg += loss.item()
                    if isinstance(loss_fn, SegmentationLoss):
                        nll_loss_avg += loss_fn.nll_loss.item()
                        center_loss_avg += loss_fn.center_loss.item()
                        smooth_loss_avg += loss_fn.smooth_loss.item()
                    elif isinstance(loss_fn, FlowLoss):
                        instance_loss_avg += loss_fn.instance_loss.item()
                        ssim_loss_avg += loss_fn.ssim_loss.item()
                        smooth_loss_avg += loss_fn.smooth_loss.item()

                    if ((batch_id % log_interval == (log_interval - 1)) or
                            (batch_id == (len(loader) - 1))):
                        loss_avg /= count
                        if isinstance(loss_fn, SegmentationLoss):
                            nll_loss_avg /= count
                            center_loss_avg /= count
                            smooth_loss_avg /= count
                        elif isinstance(loss_fn, FlowLoss):
                            instance_loss_avg /= count
                            ssim_loss_avg /= count
                            smooth_loss_avg /= count
                        completed = min(
                            len(loader.dataset),
                            (batch_id + 1) * loader.batch_size * world_size
                        )
                        msg = (
                            'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.
                            format(
                                epoch,
                                completed,
                                len(loader.dataset),
                                100. * completed / len(loader.dataset),
                                loss_avg
                            )
                        )
                        # SegmentatioinLoss only
                        if isinstance(loss_fn, SegmentationLoss):
                            msg += f'\tNLL Loss: {nll_loss_avg:.6f}'
                            msg += f'\tCenter Dice Loss: {center_loss_avg:.6f}'
                            msg += f'\tSmooth Loss: {smooth_loss_avg:.6f}'
                        # FlowLoss only
                        elif isinstance(loss_fn, FlowLoss):
                            msg += f'\tInstance Loss: {instance_loss_avg:.6f}'
                            msg += f'\tSSIM Loss: {ssim_loss_avg:.6f}'
                            msg += f'\tSmooth Loss: {smooth_loss_avg:.6f}'

                        # log to console
                        logger().info(msg)

                        # log to tensorboard
                        if log_dir is not None:
                            if 'tb_logger' not in locals():
                                tb_logger = TensorBoard(log_dir)
                            step = step_offset + epoch * len(loader) + batch_id
                            tb_logger.log_scalar(
                                tag='train_loss',
                                value=loss_avg,
                                step=step
                            )
                            # SegmentatioinLoss only
                            if isinstance(loss_fn, SegmentationLoss):
                                tb_logger.log_scalar(
                                    tag='nll_loss',
                                    value=nll_loss_avg,
                                    step=step
                                )
                                tb_logger.log_scalar(
                                    tag='center_dice_loss',
                                    value=center_loss_avg,
                                    step=step
                                )
                                tb_logger.log_scalar(
                                    tag='smooth_loss',
                                    value=smooth_loss_avg,
                                    step=step
                                )
                            # FlowLoss only
                            elif isinstance(loss_fn, FlowLoss):
                                tb_logger.log_scalar(
                                    tag='instance_loss',
                                    value=instance_loss_avg,
                                    step=step
                                )
                                tb_logger.log_scalar(
                                    tag='SSIM_loss',
                                    value=ssim_loss_avg,
                                    step=step
                                )
                                tb_logger.log_scalar(
                                    tag='smooth_loss',
                                    value=smooth_loss_avg,
                                    step=step
                                )
                        count = 0
                        loss_avg = 0
                        if isinstance(loss_fn, SegmentationLoss):
                            nll_loss_avg = 0
                            center_loss_avg = 0
                            smooth_loss_avg = 0
                        elif isinstance(loss_fn, FlowLoss):
                            instance_loss_avg = 0
                            ssim_loss_avg = 0
                            smooth_loss_avg = 0
                if is_ncrops_updated:
                    loader.dataset.length = int(
                        redis_client.get(REDIS_KEY_NCROPS))
                    loader = to_ddp_loader(loader,
                                           world_size,
                                           rank_or_device,
                                           True)
                    if eval_loader:
                        eval_loader = to_ddp_loader(eval_loader,
                                                    world_size,
                                                    rank_or_device,
                                                    False)
                    break
                if eval_loader:
                    model_eval_loss = evaluate(model,
                                               device,
                                               eval_loader,
                                               loss_fn=loss_fn,
                                               patch_size=patch_size,
                                               is_ddp=is_ddp,
                                               is_logging=is_logging)
                    if is_logging:
                        eval_loss += model_eval_loss
            if is_ncrops_updated:
                continue
            if is_logging:
                # https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/4
                state_dicts = [
                    OrderedDict({
                        k[7:]: v for k, v in model.state_dict().items()
                    }) if is_ddp else
                    model.state_dict() for model in models]
                torch.save(state_dicts[0] if len(models) == 1 else
                           state_dicts,
                           model_path)
                if eval_loader:
                    eval_loss /= len(models)

                    # log to console
                    msg = (
                        'Eval Epoch: {} \tLoss: {:.6f}'.format(
                            epoch,
                            loss
                        )
                    )
                    logger().info(msg)

                    # log to tensorboard
                    if tb_logger is not None:
                        tb_logger.log_scalar(
                            tag='eval_loss',
                            value=loss,
                            step=epoch
                        )

                    if eval_loss < min_loss:
                        min_loss = eval_loss
                        state_dicts_best = state_dicts
                        torch.save(state_dicts_best[0] if len(models) == 1 else
                                   state_dicts_best,
                                   model_path.replace('.pth', '_best.pth'))
                    else:
                        pass
                        # for model, sdict in zip(models, state_dicts_best):
                        #     model.load_state_dict(sdict)
                publish_mq('update', 'Model updated')
            epoch += 1
    finally:
        cleanup()
        torch.cuda.empty_cache()


def to_ddp_loader(loader, num_replicas, rank, shuffle):
    sampler = du.DistributedSampler(loader.dataset,
                                    num_replicas=num_replicas,
                                    rank=rank,
                                    shuffle=shuffle)
    return du.DataLoader(loader.dataset,
                         batch_size=loader.batch_size,
                         sampler=sampler,
                         num_workers=0)


def evaluate(model, device, loader, loss_fn, patch_size=None, is_ddp=False,
             is_logging=True):
    model.eval()
    model.to(device)
    loss_fn.to(device)
    loss = 0
    with torch.no_grad():
        for (x, keep_axials), y in loader:
            x, y = x.to(device), y.to(device)
            if patch_size is None:
                prediction = model(x, keep_axials)
            else:
                prediction = torch.from_numpy(
                    predict(model, device, x, keep_axials, patch_size,
                            is_logarithm=False, is_logging=is_logging,
                            batch_size=1, is_dp=False)
                ).to(device)
            loss += loss_fn(prediction, y).item()
        loss /= len(loader)

    if is_ddp:
        loss = torch.tensor(loss, device=device)
        dist.all_reduce(loss)
        loss = loss.item()
    return loss


@profile
def predict(model, device, input, keep_axials, patch_size=None,
            is_logarithm=True, is_logging=True, batch_size=1, is_dp=True,
            memmap_dir=None):
    """
    input: 5D tensor
    """
    if patch_size is None:
        patch_size = input.shape[2:]
    n_dims = len(patch_size)
    input_shape = input.shape[-n_dims:]
    n_data = input.shape[0]
    # all elements in patch shape should be smaller than or equal to those of
    # input shape
    patch_size = [min(patch_size[i], input_shape[i]) for i in range(n_dims)]
    n_patches = [
        1 if input_shape[i] == patch_size[i] else
        input_shape[i] // patch_size[i] + 1
        for i in range(n_dims)
    ]
    intervals = [
        0 if n_patches[i] == 1 else
        (
            patch_size[i] -
            (patch_size[i] * n_patches[i] - input_shape[i])
            // (n_patches[i] - 1)
        )
        for i in range(n_dims)
    ]
    overlaps = [max(1, patch_size[i] - intervals[i]) for i in range(n_dims)]
    out_channels = (
        model.module.out_conv.out_channels if isinstance(model, DDP) else
        model.out_conv.out_channels
    )
    sum_patches = n_data * np.prod(n_patches)
    i_patch = 0
    patch_list = []
    for iz in range(n_patches[-3] if n_dims == 3 else 1):
        if n_dims == 3:
            slice_z = slice(intervals[-3] * iz,
                            intervals[-3] * iz + patch_size[-3])
            if input_shape[-3] < slice_z.stop:
                slice_z = slice(input_shape[-3] - patch_size[-3],
                                input_shape[-3])
            patch_weight_z = torch.ones(patch_size,
                                        dtype=torch.float32,
                                        device=device)
            if 0 < iz:
                patch_weight_z[:overlaps[-3]] *= torch.linspace(
                    0, 1, overlaps[-3], device=device
                ).reshape(-1, 1, 1)
            if iz < n_patches[-3] - 1:
                patch_weight_z[-overlaps[-3]:] *= torch.linspace(
                    1, 0, overlaps[-3], device=device
                ).reshape(-1, 1, 1)
        for iy in range(n_patches[-2]):
            slice_y = slice(intervals[-2] * iy,
                            intervals[-2] * iy + patch_size[-2])
            if input_shape[-2] < slice_y.stop:
                slice_y = slice(input_shape[-2] - patch_size[-2],
                                input_shape[-2])
            if n_dims == 3:
                patch_weight_zy = patch_weight_z.clone()
            else:
                patch_weight_zy = torch.ones(patch_size,
                                             dtype=torch.float32,
                                             device=device)[None]
            if 0 < iy:
                patch_weight_zy[:, :overlaps[-2]] *= torch.linspace(
                    0, 1, overlaps[-2], device=device
                ).reshape(1, -1, 1)
            if iy < n_patches[-2] - 1:
                patch_weight_zy[:, -overlaps[-2]:] *= torch.linspace(
                    1, 0, overlaps[-2], device=device
                ).reshape(1, -1, 1)
            for ix in range(n_patches[-1]):
                i_patch += 1
                slice_x = slice(intervals[-1] * ix,
                                intervals[-1] * ix + patch_size[-1])
                if input_shape[-1] < slice_x.stop:
                    slice_x = slice(input_shape[-1] - patch_size[-1],
                                    input_shape[-1])
                patch_weight_zyx = patch_weight_zy.clone()
                if 0 < ix:
                    patch_weight_zyx[:, :, :overlaps[-1]] *= torch.linspace(
                        0, 1, overlaps[-1], device=device
                    ).reshape(1, 1, -1)
                if ix < n_patches[-1] - 1:
                    patch_weight_zyx[:, :, -overlaps[-1]:] *= torch.linspace(
                        1, 0, overlaps[-1], device=device
                    ).reshape(1, 1, -1)
                slices = [slice(None), slice_y, slice_x]
                if n_dims == 3:
                    slices.insert(1, slice_z)
                if n_dims == 2:
                    patch_weight_zyx = patch_weight_zyx[0]
                patch_list.append((slices, patch_weight_zyx))
    if memmap_dir:
        prediction = np.memmap(str(Path(memmap_dir) / '_prediction.dat'),
                               dtype='float32',
                               mode='w+',
                               shape=(n_data, out_channels,) + input_shape)
    else:
        prediction = np.zeros((n_data, out_channels,) + input_shape,
                              dtype='float32')
    dataset = DatasetPrediction(input,
                                patch_list,
                                keep_axials)
    loader = du.DataLoader(dataset, batch_size=batch_size,
                           shuffle=False, num_workers=0)
    if is_dp and str(device) != 'cpu' and 1 < torch.cuda.device_count():
        model = torch.nn.DataParallel(model)
    model.to(device)
    model.eval()
    with torch.no_grad():
        try:
            for batch, (x, kas, data_inds, patch_inds) in enumerate(loader):
                if (redis_client is not None and
                        get_state() == TrainState.IDLE.value):
                    raise KeyboardInterrupt
                if is_logging:
                    current_patch = min((batch + 1) * batch_size, sum_patches)
                    logger().info(
                        f'processing {current_patch} / {sum_patches}')
                x = x.to(device)
                y = model(x, kas).detach()
                if is_logarithm:
                    torch.exp_(y)
                for index in range(len(y)):
                    slices, patch_weight = patch_list[patch_inds[index]]
                    prediction[(data_inds[index],) + tuple(slices)] += (
                        (y[index] * patch_weight).cpu().numpy()
                    )
        except FileNotFoundError:
            raise KeyboardInterrupt
    return prediction


def _get_prediction(img, models, keep_axials, device, is_logarithm,
                    patch_size=None, crop_box=None, is_logging=True,
                    batch_size=1, memmap_dir=None):
    """
    input: 4D (N, C, H, W) or 5D (N, C, D, H, W) tensor
    """
    if crop_box is not None:
        slices = (slice(crop_box[1], crop_box[1] + crop_box[4]),  # Y
                  slice(crop_box[2], crop_box[2] + crop_box[5]))  # X
        if img.ndim == 5:  # Z
            slices = (slice(crop_box[0], crop_box[0] + crop_box[3]),) + slices
        slices = (slice(None), slice(None)) + slices
        img = img[slices]
    with torch.no_grad():
        x = torch.from_numpy(img)
        keep_axials = torch.tensor(keep_axials)[None]
        if len(models) == 1:
            prediction = predict(models[0],
                                 device,
                                 x,
                                 keep_axials,
                                 patch_size,
                                 is_logarithm=is_logarithm,
                                 is_logging=is_logging,
                                 batch_size=batch_size,
                                 memmap_dir=memmap_dir)[0]
        else:
            prediction = np.mean([predict(model,
                                          device,
                                          x,
                                          keep_axials,
                                          patch_size,
                                          is_logarithm=is_logarithm,
                                          is_logging=is_logging,
                                          batch_size=batch_size,
                                          memmap_dir=memmap_dir)[0]
                                  for model in models], axis=0)
    return prediction


def _get_seg_prediction(img, models, keep_axials, device, patch_size=None,
                        crop_box=None, is_logging=True, batch_size=1,
                        memmap_dir=None):
    """
    input: 4D (N, C, H, W) or 5D (N, C, D, H, W) tensor
    """
    prediction = _get_prediction(
        img, models, keep_axials, device, True, patch_size, crop_box,
        is_logging, batch_size, memmap_dir
    )
    if img.ndim == 3:
        for z in range(prediction.shape[1]):
            post_fg = np.maximum(
                prediction[2, z] - normalize_zero_one(
                    prewitt(gaussian(prediction[0, z], sigma=3))),
                0
            )
            prediction[0, z] = np.minimum(
                prediction[0, z] + (prediction[2, z] - post_fg),
                1
            )
            prediction[2, z] = post_fg
            prediction[1, z] = 1. - (prediction[0, z] + prediction[2, z])
    return prediction


def _region_to_spot(min_area, scales, origins, n_dims, use_2d, c_ratio, idx,
                    r_min, r_max, i_frame, props):
    area, centroid, bbox, image = props
    if area < min_area:
        # print('skip a spot with volume {} below threshold {}'
        #       .format(area, min_area))
        return None
    centroid = [(o + c) * s for c, s,
                o in zip(centroid, scales, origins)]
    # centroid = ((origins + centroid) * scales).tolist()
    bbox_shape = [bbox[i + n_dims] - bbox[i]
                  for i in range(n_dims)]
    # correction for floor effect duriing label generation
    c_scales = [scales[i] * (bbox_shape[i] + 1.) / bbox_shape[i]
                for i in range(n_dims)]
    if use_2d and n_dims == 3:
        c_scales[0] *= c_ratio
    moments_central = scaled_moments_central(image, c_scales, order=2)
    cov = moments_central[idx].reshape((n_dims, n_dims))
    if not cov.any():  # if all zeros
        return None
    cov /= moments_central[(0,) * n_dims]
    eigvals, eigvecs = np.linalg.eigh(cov)
    if ((eigvals < 0).any() or
        np.iscomplex(eigvals).any() or
            np.iscomplex(eigvecs).any()):
        logger().debug(
            f'skip a spot with invalid eigen value(s): {eigvals}')
        return None
    # https://stackoverflow.com/questions/22146383/covariance-matrix-of-an-ellipse
    # https://github.com/scikit-image/scikit-image/blob/master/skimage/measure/_regionprops.py#L288
    radii = 2 * np.sqrt(eigvals)
    radii /= c_ratio
    radii = np.array([radii[i] if 0 < radii[i] else max(
        r_min, scales[i]) for i in range(len(radii))])
    if (radii < r_min).any():
        # print(f'skip a spot with radii {radii} below threshold {r_min}')
        return None
    radii = np.minimum(r_max, radii)
    cov = eigvecs.dot(np.diag(radii ** 2)).dot(eigvecs.T)

    if n_dims == 2:
        centroid.insert(0, 0.)
        tmp = np.eye(3)
        tmp[-2:, -2:] = cov
        cov = tmp

    def flatten(o): return [item for sublist in o for item in sublist]
    spot = {
        't': i_frame,
        'pos': centroid[::-1],
        'covariance': flatten(cov)[::-1]
    }
    return spot


def _label_bool_inplace(image, background=None, return_num=False,
                        connectivity=None, output=None):
    """
    A modified version of skimage.measure.label, where output can be specified
    as a parameter.

    Original code:
    https://github.com/scikit-image/scikit-image/blob/e562d9f8914c804151b5247bd2e778d53510904f/skimage/measure/_label.py#L6
    """
    from skimage.morphology._util import _resolve_neighborhood
    if background == 1:
        image = ~image

    if connectivity is None:
        connectivity = image.ndim

    if not 1 <= connectivity <= image.ndim:
        raise ValueError(
            f'Connectivity for {image.ndim}D image should '
            f'be in [1, ..., {image.ndim}]. Got {connectivity}.'
        )

    footprint = _resolve_neighborhood(None, connectivity, image.ndim)
    result = ndimage.label(image, structure=footprint, output=output)
    if isinstance(result, tuple):
        output, max_label = result
    else:
        # output was written in-place
        max_label = result

    if return_num:
        return output, max_label
    else:
        return output


def _find_and_push_spots(spots, i_frame, c_probs, scales=None, c_ratio=0.4,
                         p_thresh=0.5, r_min=0, r_max=1e6, crop_box=None,
                         use_2d=False):
    n_dims = len(c_probs.shape)
    assert n_dims == 2 or n_dims == 3, (
        f'n_dims: len(c_probs.shape) shoud be 2 or 3 but got {n_dims}'
    )
    origins = crop_box[3-n_dims:3] if crop_box is not None else (0,) * n_dims
    origins = np.array(origins)

    if isinstance(c_probs, np.memmap):
        c_bin = np.memmap(c_probs.filename.replace('.dat', '_binary.dat'),
                          dtype='bool',
                          mode='w+',
                          shape=c_probs.shape)
        c_bin[:] = p_thresh < c_probs
        labels = np.memmap(c_probs.filename.replace('.dat', '_labels.dat'),
                           dtype=np.int32,
                           mode='w+',
                           shape=c_probs.shape)
        labels, max_label = _label_bool_inplace(c_bin,
                                                return_num=True,
                                                output=labels)
    else:
        labels, max_label = skimage.measure.label(p_thresh < c_probs,
                                                  return_num=True)
    regions = skimage.measure.regionprops(labels)
    logger().info(f'{max_label} detections found')

    if scales is None:
        scales = (1.,) * n_dims
    scales = np.array(scales)
    # https://forum.image.sc/t/regionprops-inertia-tensor-eigvals/23559/2
    if n_dims == 2:
        idx = ((2, 1, 1, 0),
               (0, 1, 1, 2))
        min_area = math.pi * (r_min * c_ratio) ** 2 / reduce(mul, scales)
    elif n_dims == 3:
        idx = ((2, 1, 1, 1, 0, 0, 1, 0, 0),
               (0, 1, 0, 1, 2, 1, 0, 1, 0),
               (0, 0, 1, 0, 0, 1, 1, 1, 2))
        min_area = (4 / 3 * math.pi * (r_min * c_ratio) ** 3
                    / reduce(mul, scales))

    props_list = [[region.area, region.centroid, region.bbox, region.image]
                  for region in regions]

    logger().info(
        f'calculating {"ellipse" if n_dims == 2 else "ellipsoid"} parameters...'
    )
    # multiprocessing does not help so much
    if PROFILE or True:
        results = [
            _region_to_spot(min_area, scales, origins, n_dims, use_2d, c_ratio,
                            idx, r_min, r_max, i_frame, props)
            for props in props_list
        ]
        for r in results:
            if (redis_client is not None and
                    get_state() == TrainState.IDLE.value):
                raise KeyboardInterrupt
            if r is not None:
                spots.append(r)
    else:
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.imap(
                partial(_region_to_spot, min_area, scales, origins, n_dims,
                        use_2d, c_ratio, idx, r_min, r_max, i_frame),
                props_list
            )
            for r in results:
                if r is not None:
                    spots.append(r)
    logger().info(f'{len(spots)} detections kept')


def detect_spots(device, model_path, keep_axials=(True,) * 4, is_pad=False,
                 is_3d=True, crop_size=(16, 384, 384), scales=None,
                 cache_maxbytes=None, use_2d=False, use_median=False,
                 patch_size=None, crop_box=None, c_ratio=0.4, p_thresh=0.5,
                 r_min=0, r_max=1e6, output_prediction=False, zpath_input=None,
                 zpath_seg_output=None, timepoint=None, tiff_input=None,
                 memmap_dir=None, batch_size=1, input_size=None):
    use_median = use_median and is_3d
    n_dims = 2 + is_3d
    resize_factor = [1, ] * n_dims
    if tiff_input is not None:
        img_input = skimage.io.imread(tiff_input).astype('float32')
        if use_median:
            global_median = np.median(img_input)
            for z in range(img_input.shape[0]):
                slice_median = np.median(img_input[z])
                if 0 < slice_median:
                    img_input[z] -= slice_median - global_median
        img_input = normalize_zero_one(img_input)
        if input_size is not None and img_input.shape[-n_dims:] != input_size:
            original_size = img_input.shape
            img_input = F.interpolate(torch.from_numpy(img_input)[None, None],
                                      size=input_size,
                                      mode=('trilinear' if is_3d else
                                            'bilinear'),
                                      align_corners=True,
                                      )[0, 0].numpy()
            resize_factor = [input_size[d] / original_size[d] for d in
                             range(n_dims)]
    elif zpath_input is not None:
        za_input = zarr.open(zpath_input, mode='r')
        img_input = get_input_at(za_input,
                                 timepoint,
                                 memmap_dir=memmap_dir,
                                 use_median=use_median,
                                 img_size=input_size)
        if input_size is not None and za_input.shape[-n_dims:] != input_size:
            original_size = za_input.shape[-n_dims:]
            resize_factor = [input_size[d] / original_size[d] for d in
                             range(n_dims)]
    else:
        raise RuntimeError('No image is specified.')
    if any(factor != 1 for factor in resize_factor):
        if crop_box is not None:
            original_crop_box = [elem for elem in crop_box]
        for d in range(n_dims):
            scales[-(1+d)] = scales[-(1+d)] / resize_factor[-(1+d)]
            if crop_box is not None:
                crop_box[2-d] = int(crop_box[2-d] * resize_factor[-(1+d)])
                crop_box[5-d] = int(crop_box[5-d] * resize_factor[-(1+d)])
    try:
        models = load_seg_models(model_path,
                                 keep_axials,
                                 device,
                                 is_eval=True,
                                 is_pad=is_pad,
                                 is_3d=is_3d,
                                 zpath_input=zpath_input,
                                 crop_size=crop_size,
                                 scales=scales,
                                 cache_maxbytes=cache_maxbytes)
        if len(img_input.shape) == 3 and use_2d:
            prediction = np.swapaxes(np.array([
                _get_seg_prediction(img_input[z][None, None],
                                    models,
                                    keep_axials,
                                    device,
                                    patch_size[-2:],
                                    crop_box,
                                    batch_size=batch_size,
                                    memmap_dir=memmap_dir)
                for z in range(img_input.shape[0])
            ]), 0, 1)
        else:
            prediction = _get_seg_prediction(img_input[None, None],
                                             models,
                                             keep_axials,
                                             device,
                                             patch_size,
                                             crop_box,
                                             batch_size=batch_size,
                                             memmap_dir=memmap_dir)

        # calculate spots
        spots = []
        _find_and_push_spots(spots,
                             timepoint,
                             # last channel is the center label
                             prediction[-1],
                             scales,
                             c_ratio=c_ratio,
                             p_thresh=p_thresh,
                             r_min=r_min,
                             r_max=r_max,
                             crop_box=crop_box,
                             use_2d=use_2d)

        # output prediction
        if output_prediction and zpath_seg_output is not None:
            za_seg = zarr.open(zpath_seg_output, mode='a')
            if any(factor != 1 for factor in resize_factor):
                if crop_box is not None:
                    size = original_crop_box[-n_dims:]
                else:
                    size = original_size
                prediction = F.interpolate(
                    torch.from_numpy(prediction)[None],
                    size=size,
                    mode=('trilinear' if is_3d else
                          'bilinear'),
                    align_corners=True,
                )[0].numpy()
                if crop_box is not None:
                    crop_box = original_crop_box
            slices_crop = tuple(
                slice(crop_box[i],
                      crop_box[i] + crop_box[i + 3])
                if crop_box is not None else
                slice(None)
                for i in range(0 if is_3d else 1, 3)
            )
            if is_3d:  # 3D
                dims_order = [1, 2, 3, 0]
            else:  # 2D
                # slices_crop = (0,) + slices_crop
                dims_order = [1, 2, 0]
            slices_crop = (timepoint,) + slices_crop
            za_seg[slices_crop] = (np.transpose(prediction, dims_order)
                                   .astype('float16'))
    except KeyboardInterrupt:
        logger().info('KeyboardInterrupt')
        raise KeyboardInterrupt
    finally:
        torch.cuda.empty_cache()
        if memmap_dir is not None:
            for p in Path(memmap_dir).glob('_prediction*'):
                p.unlink()

    return spots


def _get_flow_prediction(img, models, keep_axials, device, patch_size=None,
                         crop_box=None, is_logging=True, batch_size=1,
                         memmap_dir=None):
    """
    input: 4D (N, C, H, W) or 5D (N, C, D, H, W) tensor
    """
    # save and use as float16 to save the storage
    return _get_prediction(
        img, models, keep_axials, device, False, patch_size, crop_box,
        is_logging, batch_size, memmap_dir
    ).astype('float16')


def _estimate_spots_with_flow(spots, flow_stack, scales, resize_factor,
                              crop_box=None):
    img_shape = flow_stack[0].shape
    n_dims = len(img_shape)
    assert n_dims == 2 or n_dims == 3, (
        f'n_dims: len(img_shape.shape) shoud be 2 or 3 but got {n_dims}'
    )
    assert n_dims == flow_stack.shape[0], (
        f'n_dims: {n_dims} shoud be equal to '
        f'flow_stack.shape[0]: {flow_stack.shape[0]}'
    )
    origins = crop_box[3-n_dims:3] if crop_box is not None else (0,) * n_dims
    origins = np.array(origins)
    if scales is None:
        scales = (1.,) * n_dims
    scales = np.array(scales)
    resize_factor = np.array(resize_factor)
    origins = origins * scales
    MIN_AREA_ELLIPSOID = 9
    res_spots = []
    draw_func = ellipsoid if n_dims == 3 else ellipse
    for spot in spots:
        if (redis_client is not None and
                get_state() == TrainState.IDLE.value):
            raise KeyboardInterrupt
        spot_id = spot['id']
        pos = spot['pos']  # XYZ
        centroid = np.array(pos[::-1])  # ZYX
        centroid = centroid[-n_dims:]
        centroid = centroid - origins
        covariance = np.array(spot['covariance'][::-1]).reshape(3, 3)
        covariance = covariance[-n_dims:, -n_dims:]
        radii, rotation = np.linalg.eigh(covariance)
        radii = np.sqrt(radii)
        indices = draw_func(centroid,
                            radii,
                            rotation,
                            scales,
                            img_shape,
                            MIN_AREA_ELLIPSOID)
        if 0 < len(indices[0]):
            displacement = [
                flow_stack[to_fancy_index(dim, *indices)].mean()
                * scales[-1 - dim] * resize_factor[-1 - dim]
                for dim in range(n_dims)
            ]
            res_spots.append(
                {
                    'id': spot_id,
                    'pos': [
                        pos[dim] + displacement[dim]
                        for dim in range(n_dims)
                    ] + ([0, ] if n_dims == 2 else []),
                    'sqdisp': sum([
                        displacement[dim]**2
                        for dim in range(n_dims)
                    ]),
                }
            )
        else:
            res_spots.append(
                {
                    'id': spot_id,
                    'pos': pos,
                    'sqdisp': 0,
                }
            )
    return res_spots


def spots_with_flow(device, spots, model_path, keep_axials=(True,) * 4,
                    is_pad=False, is_3d=True, scales=None, use_median=False,
                    patch_size=None, crop_box=None, output_prediction=False,
                    zpath_input=None, zpath_flow=None, timepoint=None,
                    tiff_input=None, memmap_dir=None, batch_size=1,
                    input_size=None, flow_norm_factor=None):
    use_median = use_median and is_3d
    n_dims = 2 + is_3d
    resize_factor = [1, ] * n_dims
    if tiff_input is not None:
        img_input = np.array(
            [normalize_zero_one(skimage.io.imread(f).astype('float32'))
             for f in tiff_input]
        )
        if img_input.shape[-n_dims:] != input_size:
            original_size = img_input.shape
            img_input = F.interpolate(torch.from_numpy(img_input)[None],
                                      size=input_size,
                                      mode=('trilinear' if is_3d else
                                            'bilinear'),
                                      align_corners=True,
                                      )[0].numpy()
            resize_factor = [img_input.shape[d] / original_size[d] for d in
                             range(n_dims)]
    elif zpath_input is not None:
        za_input = zarr.open(zpath_input, mode='r')
        img_input = get_inputs_at(za_input,
                                  timepoint - 1,
                                  memmap_dir=memmap_dir,
                                  img_size=input_size)
        if input_size is not None and za_input.shape[-n_dims:] != input_size:
            original_size = za_input.shape[-n_dims:]
            resize_factor = [input_size[d] / original_size[d] for d in
                             range(n_dims)]
    if any(factor != 1 for factor in resize_factor):
        if crop_box is not None:
            original_crop_box = [elem for elem in crop_box]
        for d in range(n_dims):
            scales[-(1+d)] = scales[-(1+d)] / resize_factor[-(1+d)]
            if crop_box is not None:
                crop_box[2-d] = int(crop_box[2-d] * resize_factor[-(1+d)])
                crop_box[5-d] = int(crop_box[5-d] * resize_factor[-(1+d)])
    try:
        models = load_flow_models(model_path,
                                  device,
                                  is_eval=True,
                                  is_pad=is_pad,
                                  is_3d=is_3d)
        prediction = _get_flow_prediction(img_input[None],
                                          models,
                                          keep_axials,
                                          device,
                                          patch_size,
                                          crop_box,
                                          batch_size=batch_size,
                                          memmap_dir=memmap_dir)

        if flow_norm_factor is not None:
            # Restore to voxel unit
            for d in range(prediction.shape[0]):
                prediction[d] *= flow_norm_factor[d]
        res_spots = _estimate_spots_with_flow(spots,
                                              prediction,
                                              scales,
                                              resize_factor,
                                              crop_box)
    finally:
        torch.cuda.empty_cache()
    if output_prediction and zpath_flow is not None:
        za_flow = zarr.open(zpath_flow, mode='a')
        if any(factor != 1 for factor in resize_factor):
            if crop_box is not None:
                size = original_crop_box[-n_dims:]
            else:
                size = original_size
            prediction = F.interpolate(
                torch.from_numpy(prediction)[None].type(torch.float32),
                size=size,
                mode=('trilinear' if is_3d else
                      'bilinear'),
                align_corners=True,
            )[0].numpy()
            if crop_box is not None:
                crop_box = original_crop_box
        slices_crop = tuple(
            slice(crop_box[i],
                  crop_box[i] + crop_box[i + 3])
            if crop_box is not None else
            slice(None)
            for i in range(0 if is_3d else 1, 3)
        )
        slices_crop = (timepoint - 1, slice(None)) + slices_crop
        if flow_norm_factor is not None:
            # Normalize
            for d in range(prediction.shape[0]):
                prediction[d] /= flow_norm_factor[d]
        za_flow[slices_crop] = prediction.astype('float16')
    return res_spots


def export_ctc_labels(config, spots_dict):
    MIN_AREA_ELLIPSOID = 20
    timepoints = set([*range(config.t_start, config.t_end + 1, 1)])
    is_zip = config.savedir is None
    savedir = tempfile.mkdtemp() if is_zip else config.savedir
    n_dims = len(config.shape)
    digits = max(3, len(str(max(list(spots_dict.keys())))))
    for t, spots in spots_dict.items():
        if (redis_client is not None and
                get_state() == TrainState.IDLE.value):
            logger().info('aborted')
            if is_zip:
                shutil.rmtree(savedir)
            return False
        label = np.zeros(config.shape, np.uint16)
        for i in range(16):
            for spot in spots:
                centroid = np.array(spot['pos'][::-1])
                centroid = centroid[-n_dims:]
                covariance = np.array(spot['covariance'][::-1]).reshape(3, 3)
                covariance = covariance[-n_dims:, -n_dims:]
                radii, rotation = np.linalg.eigh(covariance)
                radii = np.sqrt(radii)
                draw_func = ellipsoid if n_dims == 3 else ellipse
                indices = draw_func(
                    centroid,
                    radii * (1 - 0.05 * i),
                    rotation,
                    config.scales,
                    label.shape[-n_dims:],
                    MIN_AREA_ELLIPSOID
                )
                label[indices] = spot['value']
        # ensure that each spot is labeled at least its center voxcel
        for spot in spots:
            centroid = np.array(spot['pos'][::-1])
            centroid = centroid[-n_dims:]
            indices_center = tuple(int(centroid[i] / config.scales[i])
                                   for i in range(n_dims))
            label[indices_center] = spot['value']
        skimage.io.imsave(
            os.path.join(savedir, f'mask{t:0{digits}d}.tif'),
            label[:, None] if n_dims == 3 else label,
            imagej=True,
            compress=6,
        )
        timepoints.discard(t)
    # Save blank images for unlabeled frames
    label = np.zeros(config.shape, np.uint16)
    for t in timepoints:
        skimage.io.imsave(
            os.path.join(savedir, f'mask{t:0{digits}d}.tif'),
            label[:, None] if n_dims == 3 else label,
            imagej=True,
            compress=6,
        )
    if is_zip:
        zip_path = shutil.make_archive('/tmp/archive', 'zip', savedir)
        shutil.rmtree(savedir)
        return zip_path
    return True


def init_seg_models(model_path, keep_axials, device, is_3d=True, n_models=1,
                    n_crops=0, zpath_input=None, crop_size=None, scales=None,
                    url='Versatile', state_dicts=None, is_cpu=False,
                    cache_maxbytes=None):
    if state_dicts is None:
        state_dicts = [None, ] * n_models
        if url == 'Versatile':
            url = f'{MODEL_URL_ROOT}versatile{2+is_3d}d.pth'
        if url is not None:
            logger().info(f'Loading a pretrained model: {url}')
            checkpoint = torch.hub.load_state_dict_from_url(
                url, progress=False, map_location=device,
            )
            state_dicts = checkpoint if isinstance(
                checkpoint, list) else [checkpoint]
    models = [UNet.three_class_segmentation(
        device=device,
        is_3d=is_3d,
        state_dict=state_dicts[i],
    ) for i in range(n_models)]
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(models[0].state_dict() if len(models) == 1 else
               [model.state_dict() for model in models], model_path)
    if all(state_dict is None for state_dict in state_dicts):
        if 0 < n_crops:
            errors = []
            if zpath_input is None:
                errors.append('zpath_input is required. Skip prior training.')
            if crop_size is None:
                errors.append('crop_size is required. Skip prior training.')
            if 0 < len(errors):
                for error in errors:
                    logger().error(error)
            else:
                for i, lr in enumerate([1e-2, 1e-3, 1e-4]):
                    run_train_prior_seg(
                        device, 1, crop_size, model_path, 1, keep_axials,
                        scales, lr, n_crops, is_3d, zpath_input,
                        log_interval=1, epoch_start=i, is_cpu=is_cpu,
                        cache_maxbytes=cache_maxbytes,
                    )


def init_flow_models(model_path, device, is_3d=True, n_models=1,
                     url='Versatile', state_dicts=None):
    if state_dicts is None:
        state_dicts = [None, ] * n_models
        if url == 'Versatile':
            if is_3d:
                url = f'{MODEL_URL_ROOT}Fluo-N3DH-CE_flow.pth'
            else:
                logger().info(
                    'The 2D versatile model is not available, ' +
                    'initialize a model with random parameters.'
                )
                url = None
        if url is not None:
            logger().info(f'Loading a pretrained model: {url}')
            checkpoint = torch.hub.load_state_dict_from_url(
                url, progress=False, map_location=device,
            )
            state_dicts = checkpoint if isinstance(
                checkpoint, list) else [checkpoint]
    models = [FlowResNet.three_dimensional_flow(
        device=device,
        is_3d=is_3d,
        state_dict=state_dicts[i],
    ) for i in range(n_models)]
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(models[0].state_dict() if len(models) == 1 else
               [model.state_dict() for model in models], model_path)


def load_seg_models(model_path, keep_axials, device, is_eval=False,
                    is_decoder_only=False, is_pad=False, is_3d=True,
                    n_models=1, n_crops=5, zpath_input=None, crop_size=None,
                    scales=None, url='Versatile', is_cpu=False,
                    cache_maxbytes=None):
    if not Path(model_path).exists():
        logger().info(
            f'Model file {model_path} not found. Start initialization...')
        try:
            init_seg_models(model_path,
                            keep_axials,
                            device,
                            is_3d,
                            n_models,
                            n_crops,
                            zpath_input,
                            crop_size,
                            scales,
                            url=url,
                            is_cpu=is_cpu,
                            cache_maxbytes=cache_maxbytes)
        finally:
            gc.collect()
            torch.cuda.empty_cache()
    checkpoint = torch.load(model_path, map_location=device)
    state_dicts = checkpoint if isinstance(
        checkpoint, list) else [checkpoint]
    # print(len(state_dicts), 'models will be ensembled')
    models = [UNet.three_class_segmentation(
        is_eval=is_eval,
        device=device,
        state_dict=state_dict,
        is_decoder_only=is_decoder_only,
        is_pad=is_pad,
        is_3d=is_3d,
    ) for state_dict in state_dicts]
    return models


def load_flow_models(model_path, device, is_eval=False,
                     is_decoder_only=False, is_pad=False, is_3d=True,
                     n_models=1, url='Versatile'):
    if not Path(model_path).exists():
        logger().info(
            f'Model file {model_path} not found. Start initialization...')
        init_flow_models(model_path,
                         device,
                         is_3d,
                         n_models,
                         url=url)
    checkpoint = torch.load(model_path, map_location=device)
    state_dicts = checkpoint if isinstance(
        checkpoint, list) else [checkpoint]
    # print(len(state_dicts), 'models will be ensembled')
    models = [FlowResNet.three_dimensional_flow(
        is_eval=is_eval,
        device=device,
        state_dict=state_dict,
        is_decoder_only=is_decoder_only,
        is_pad=is_pad,
        is_3d=is_3d,
    ) for state_dict in state_dicts]
    return models
