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

import hashlib
from functools import reduce
import json
import math
from operator import mul
import os
import shutil
import tempfile
import time
import traceback

import numpy as np
from skimage.filters import gaussian
from skimage.filters import prewitt
import skimage.io
import skimage.measure
import torch
import torch.utils.data as du
if os.environ.get('CTC') != '1':
    import tensorboardX as tb
    import zarr

from elephant.datasets import AutoencoderDatasetZarr
from elephant.losses import AutoencoderLoss
from elephant.models import FlowResNet
from elephant.models import UNet
from elephant.models import load_flow_models
from elephant.models import load_seg_models
from elephant.redis_util import REDIS_KEY_LR
from elephant.redis_util import REDIS_KEY_STATE
from elephant.redis_util import TrainState
from elephant.util import get_pad_size
from elephant.util import normalize_zero_one
from elephant.util.ellipse import ellipse
from elephant.util.ellipsoid import ellipsoid
from elephant.util.scaled_moments import scaled_moments_central


class TensorBoard(object):
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.writer = tb.SummaryWriter(self.log_dir)

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag=tag, scalar_value=value, global_step=step)


def train(model, device, loader, optimizer, loss_fn, epoch,
          log_interval=100, tb_logger=None, redis_client=None):
    model.train()
    model.to(device)
    for batch_id, (x, y) in enumerate(loader):
        # (torch.tensor(-100.), torch.tensor(-100)) is returned when aborted
        if (torch.eq(x, torch.tensor(-100.)).any() and
                torch.eq(y, torch.tensor(-100)).any()):
            break
        if redis_client is not None:
            while (int(redis_client.get(REDIS_KEY_STATE)) ==
                   TrainState.WAIT.value):
                print("waiting")
                time.sleep(1)
            if (int(redis_client.get(REDIS_KEY_STATE)) ==
                    TrainState.IDLE.value):
                break

        try:
            x, y = x.to(device), y.to(device)

            # set leargning rate
            if (redis_client is not None and
                    redis_client.get(REDIS_KEY_LR) is not None):
                for g in optimizer.param_groups:
                    g['lr'] = float(redis_client.get(REDIS_KEY_LR))

            optimizer.zero_grad()
            prediction = model(x)
            loss = loss_fn(prediction, y)
            loss.backward()
            optimizer.step()

            # log to console
            if batch_id % log_interval == 0:
                msg = (
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch,
                        batch_id * len(x),
                        len(loader.dataset),
                        100. * batch_id / len(loader),
                        loss.item()
                    )
                )
                # SegmentatioinLoss only
                try:
                    msg += f'\tNLL Loss: {loss_fn.nll_loss:.6f}'
                    msg += f'\tCenter Dice Loss: {loss_fn.center_loss:.6f}'
                    msg += f'\tSmooth Loss: {loss_fn.smooth_loss:.6f}'
                except AttributeError:
                    pass
                # FlowLoss only
                try:
                    msg += f'\tInstance Loss: {loss_fn.instance_loss:.6f}'
                    msg += f'\tSSIM Loss: {loss_fn.ssim_loss:.6f}'
                    msg += f'\tSmooth Loss: {loss_fn.smooth_loss:.6f}'
                except AttributeError:
                    pass
                print(msg)

            # log to tensorboard
            if tb_logger is not None:
                step = epoch * len(loader) + batch_id
                tb_logger.log_scalar(
                    tag='train_loss',
                    value=loss.item(),
                    step=step
                )
                # SegmentatioinLoss only
                try:
                    tb_logger.log_scalar(
                        tag='nll_loss',
                        value=loss_fn.nll_loss,
                        step=step
                    )
                    tb_logger.log_scalar(
                        tag='center_dice_loss',
                        value=loss_fn.center_loss,
                        step=step
                    )
                    tb_logger.log_scalar(
                        tag='smooth_loss',
                        value=loss_fn.smooth_loss,
                        step=step
                    )
                except AttributeError:
                    pass
                # FlowLoss only
                try:
                    tb_logger.log_scalar(
                        tag='instance_loss',
                        value=loss_fn.instance_loss,
                        step=step
                    )
                    tb_logger.log_scalar(
                        tag='SSIM_loss',
                        value=loss_fn.ssim_loss,
                        step=step
                    )
                    tb_logger.log_scalar(
                        tag='smooth_loss',
                        value=loss_fn.smooth_loss,
                        step=step
                    )
                except AttributeError:
                    pass
        finally:
            torch.cuda.empty_cache()


def evaluate(model, device, loader, loss_fn, epoch, tb_logger=None):
    model.eval()
    model.to(device)
    loss = 0
    with torch.no_grad():
        for batch_id, (x, y) in enumerate(loader):
            # (torch.tensor(-100.), torch.tensor(-100))
            # is returned when aborted
            if (torch.eq(x, torch.tensor(-100.)).any() and
                    torch.eq(y, torch.tensor(-100)).any()):
                break

            x, y = x.to(device), y.to(device)
            prediction = model(x)
            loss += loss_fn(prediction, y).item()
        loss /= len(loader)
    # log to console
    msg = (
        'Eval Epoch: {} \tLoss: {:.6f}'.format(
            epoch,
            loss
        )
    )
    print(msg)

    # log to tensorboard
    if tb_logger is not None:
        tb_logger.log_scalar(
            tag='eval_loss',
            value=loss,
            step=epoch
        )
    return loss


def _patch_predict(model, x, patch_size, func):
    n_dims = len(patch_size)
    input_shape = x.shape[-n_dims:]
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
    prediction = np.zeros((3,) + input_shape, dtype='float32')
    for iz in range(n_patches[-3] if n_dims == 3 else 1):
        if n_dims == 3:
            slice_z = slice(intervals[-3] * iz,
                            intervals[-3] * iz + patch_size[-3])
            if x.shape[-3] < slice_z.stop:
                slice_z = slice(x.shape[-3] - patch_size[-3], x.shape[-3])
            patch_weight_z = np.ones(patch_size, dtype='float32')
            if 0 < iz:
                patch_weight_z[:overlaps[-3]] *= np.linspace(
                    0, 1, overlaps[-3]
                ).reshape(-1, 1, 1)
            if iz < n_patches[-3] - 1:
                patch_weight_z[-overlaps[-3]:] *= np.linspace(
                    1, 0, overlaps[-3]
                ).reshape(-1, 1, 1)
        for iy in range(n_patches[-2]):
            slice_y = slice(intervals[-2] * iy,
                            intervals[-2] * iy + patch_size[-2])
            if x.shape[-2] < slice_y.stop:
                slice_y = slice(x.shape[-2] - patch_size[-2],
                                x.shape[-2])
            if n_dims == 3:
                patch_weight_zy = patch_weight_z.copy()
            else:
                patch_weight_zy = np.ones(patch_size, dtype='float32')[None]
            if 0 < iy:
                patch_weight_zy[:, :overlaps[-2]] *= np.linspace(
                    0, 1, overlaps[-2]
                ).reshape(1, -1, 1)
            if iy < n_patches[-2] - 1:
                patch_weight_zy[:, -overlaps[-2]:] *= np.linspace(
                    1, 0, overlaps[-2]
                ).reshape(1, -1, 1)
            for ix in range(n_patches[-1]):
                slice_x = slice(intervals[-1] * ix,
                                intervals[-1] * ix + patch_size[-1])
                if x.shape[-1] < slice_x.stop:
                    slice_x = slice(x.shape[-1] - patch_size[-1], x.shape[-1])
                patch_weight_zyx = patch_weight_zy.copy()
                if 0 < ix:
                    patch_weight_zyx[:, :, :overlaps[-1]] *= np.linspace(
                        0, 1, overlaps[-1]
                    ).reshape(1, 1, -1)
                if ix < n_patches[-1] - 1:
                    patch_weight_zyx[:, :, -overlaps[-1]:] *= np.linspace(
                        1, 0, overlaps[-1]
                    ).reshape(1, 1, -1)
                slices = [slice(None), slice(None), slice_y, slice_x]
                if n_dims == 3:
                    slices.insert(2, slice_z)
                y = func(model(x[slices])[0].detach().cpu().numpy())
                if n_dims == 2:
                    patch_weight_zyx = patch_weight_zyx[0]
                prediction[slices[1:]] += y * patch_weight_zyx
    return prediction


def predict(model, x, patch_size=None, is_log=True):
    func = (lambda x: np.exp(x)) if is_log else (lambda x: x)
    if patch_size is not None:
        return _patch_predict(model, x, patch_size, func)
    else:
        return func(model(x)[0].detach().cpu().numpy())


def _get_seg_prediction(img, timepoint, model_path, keep_axials, device,
                        use_median=True, patch_size=None, crop_box=None,
                        is_pad=False):
    img = img.astype('float32')
    n_dims = len(img.shape)
    is_3d = n_dims == 3
    if use_median:
        global_median = np.median(img)
        for z in range(img.shape[0]):
            slice_median = np.median(img[z])
            if 0 < slice_median:
                img[z] -= slice_median - global_median
    img = normalize_zero_one(img)
    models = load_seg_models(model_path, keep_axials, device, is_eval=True,
                             is_pad=is_pad, is_3d=is_3d)
    if crop_box is not None:
        slices = (slice(crop_box[1], crop_box[1] + crop_box[4]),  # Y
                  slice(crop_box[2], crop_box[2] + crop_box[5]))  # X
        if is_3d:  # Z
            slices = (slice(crop_box[0], crop_box[0] + crop_box[3]),) + slices
        img = img[slices]
    if patch_size is None:
        pad_base = (16, 16)
        if is_3d:
            pad_base = (2**keep_axials.count(False),) + pad_base
        pad_shape = tuple(
            get_pad_size(img.shape[i], pad_base[i])
            for i in range(n_dims)
        )
        slices = (slice(None),) + tuple(
            slice(None) if max(pad_shape[i]) == 0 else
            slice(pad_shape[i][0], -pad_shape[i][1])
            for i in range(n_dims)
        )
        img = np.pad(img, pad_shape, mode='constant', constant_values=0)
    with torch.no_grad():
        x = torch.from_numpy(img[np.newaxis, np.newaxis]).to(device)
        prediction = np.mean([predict(model, x, patch_size, is_log=True)
                              for model in models], axis=0)
    if patch_size is None:
        prediction = prediction[slices]
    if is_3d:
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


def detect_spots(config):
    if hasattr(config, 'tiff_input') and config.tiff_input is not None:
        img_input = skimage.io.imread(config.tiff_input)
    elif config.zpath_input is not None:
        za_input = zarr.open(config.zpath_input, mode='a')
        img_input = za_input[config.timepoint]
    try:
        prediction = _get_seg_prediction(img_input,
                                         config.timepoint,
                                         config.model_path,
                                         config.keep_axials,
                                         config.device,
                                         config.use_median,
                                         config.patch_size,
                                         config.crop_box,
                                         config.is_pad)
    except Exception:
        print(traceback.format_exc())
    finally:
        torch.cuda.empty_cache()

    spots = []
    _find_and_push_spots(spots,
                         config.timepoint,
                         prediction[-1],  # last channel is the center label
                         config.scales,
                         c_ratio=config.c_ratio,
                         p_thresh=config.p_thresh,
                         r_min=config.r_min,
                         r_max=config.r_max,
                         crop_box=config.crop_box)

    if config.output_prediction and config.zpath_seg_output is not None:
        za_seg = zarr.open(config.zpath_seg_output, mode='a')
        slices_crop = tuple(
            slice(config.crop_box[i],
                  config.crop_box[i] + config.crop_box[i + 3])
            if config.crop_box is not None else
            slice(None)
            for i in range(0 if config.is_3d else 1, 3)
        )
        za_seg[config.timepoint][slices_crop] = (np.transpose(prediction,
                                                              (1, 2, 3, 0))
                                                 .astype('float16'))

    return spots


def _get_flow_prediction(img, timepoint, model_path, keep_axials, device,
                         flow_norm_factor, patch_size, is_3d):
    img = normalize_zero_one(img.astype('float32'))
    models = load_flow_models(
        model_path, keep_axials, device, is_eval=True, is_3d=is_3d)

    if patch_size is None:
        pad_base = (2**keep_axials.count(False), 16, 16)
        pad_shape = tuple(
            get_pad_size(img.shape[i + 1], pad_base[i])
            for i in range(len(img.shape) - 1)
        )
        slices = (slice(None),) + tuple(
            slice(None) if max(pad_shape[i]) == 0 else
            slice(pad_shape[i][0], -pad_shape[i][1])
            for i in range(len(pad_shape))
        )
        img = np.pad(img, ((0, 0),) + pad_shape, mode='reflect')
    with torch.no_grad():
        x = torch.from_numpy(img[np.newaxis]).to(device)
        prediction = np.mean(
            [
                predict(model, x, patch_size, is_log=False)
                for model in models
            ],
            axis=0)
    if patch_size is None:
        prediction = prediction[slices]
    # save and use as float16 to save the storage
    return prediction.astype('float16')


def _estimate_spots_with_flow(spots, flow_stack, scales):
    img_shape = flow_stack[0].shape
    MIN_AREA_ELLIPSOID = 9
    res_spots = []
    for spot in spots:
        spot_id = spot['id']
        pos = spot['pos']  # XYZ
        centroid = np.array(pos[::-1])  # ZYX
        covariance = np.array(spot['covariance'][::-1]).reshape(3, 3)
        radii, rotation = np.linalg.eigh(covariance)
        radii = np.sqrt(radii)
        dd, rr, cc = ellipsoid(centroid,
                               radii,
                               rotation,
                               scales,
                               img_shape,
                               MIN_AREA_ELLIPSOID)
        if 0 < len(dd):
            displacement = [
                flow_stack[dim][dd, rr, cc].mean() * scales[-1 - dim]
                for dim in range(flow_stack.shape[0])
            ]
            res_spots.append(
                {
                    'id': spot_id,
                    'pos': [
                        pos[dim] + displacement[dim]
                        for dim in range(flow_stack.shape[0])
                    ],
                    'sqdisp': sum([
                        displacement[dim]**2
                        for dim in range(flow_stack.shape[0])
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


def spots_with_flow(config, spots):
    prediction = None
    if hasattr(config, 'tiff_input') and config.tiff_input is not None:
        img_input = np.array([skimage.io.imread(f) for f in config.tiff_input])
    elif config.zpath_input is not None:
        za_input = zarr.open(config.zpath_input, mode='a')
        za_flow = zarr.open(config.zpath_flow, mode='a')
        za_hash = zarr.open(config.zpath_flow_hashes, mode='a')

        # https://stackoverflow.com/questions/3431825/generating-an-md5-checksum-of-a-file#answer-3431838
        hash_md5 = hashlib.md5()
        with open(config.model_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_md5.update(chunk)
        za_md5 = zarr.array(
            za_input[config.timepoint - 1:config.timepoint + 1]).digest('md5')
        hash_md5.update(za_md5)
        hash_md5.update(json.dumps(config.patch_size).encode('utf-8'))
        model_md5 = hash_md5.digest()
        if model_md5 == za_hash[config.timepoint - 1]:
            prediction = za_flow[config.timepoint - 1]
        else:
            img_input = np.array([
                normalize_zero_one(za_input[i].astype('float32'))
                for i in range(config.timepoint - 1, config.timepoint + 1)
            ])
    if prediction is None:
        try:
            prediction = _get_flow_prediction(img_input,
                                              config.timepoint,
                                              config.model_path,
                                              config.keep_axials,
                                              config.device,
                                              config.flow_norm_factor,
                                              config.patch_size,
                                              config.is_3d)
        finally:
            torch.cuda.empty_cache()
        if config.output_prediction:
            za_flow[config.timepoint - 1] = prediction
            za_hash[config.timepoint - 1] = model_md5
        else:
            za_hash[config.timepoint - 1] = 0
    # Restore to voxel unit
    for d in range(prediction.shape[0]):
        prediction[d] *= config.flow_norm_factor[d]
    res_spots = _estimate_spots_with_flow(spots, prediction, config.scales)
    return res_spots


def _find_and_push_spots(spots, i_frame, c_probs, scales=None, c_ratio=0.5,
                         p_thresh=0.3, r_min=0, r_max=1e6, crop_box=None):
    n_dims = len(c_probs.shape)
    assert n_dims == 2 or n_dims == 3, (
        f'n_dims: len(c_probs.shape) shoud be 2 or 3 but got {n_dims}'
    )
    origins = crop_box[3-n_dims:3] if crop_box is not None else (0,) * n_dims
    labels = skimage.measure.label(p_thresh < c_probs)
    regions = skimage.measure.regionprops(labels, c_probs)

    if scales is None:
        scales = (1.,) * n_dims
    # https://forum.image.sc/t/regionprops-inertia-tensor-eigvals/23559/2
    if n_dims == 2:
        idx = ((2, 1, 1, 0),
               (0, 1, 1, 2))
        min_area = math.pi * (r_min * c_ratio) ** 2 / reduce(mul, scales)
    elif n_dims == 3:
        idx = ((2, 1, 1, 1, 0, 0, 1, 0, 0),
               (0, 1, 0, 1, 2, 1, 0, 1, 0),
               (0, 0, 1, 0, 0, 1, 1, 1, 2))
        min_area = 4 / 3 * math.pi * \
            (r_min * c_ratio) ** 3 / reduce(mul, scales)
    for i, region in enumerate(regions):
        if region.area < min_area:
            # print('skip a spot with volume {} below threshold {}'
            #       .format(region.area, min_area))
            continue
        centroid = [(o + c) * s for c, s,
                    o in zip(region.centroid, scales, origins)]
        bbox_shape = [region.bbox[i + n_dims] - region.bbox[i]
                      for i in range(n_dims)]
        # correction for floor effect duriing label generation
        c_scales = [scales[i] * (bbox_shape[i] + 1.) / bbox_shape[i]
                    for i in range(n_dims)]
        moments_central = scaled_moments_central(
            region.image, c_scales, order=2)
        cov = moments_central[idx].reshape((n_dims, n_dims))
        if not cov.any():  # if all zeros
            continue
        cov /= moments_central[(0,) * n_dims]
        eigvals, eigvecs = np.linalg.eigh(cov)
        if ((eigvals < 0).any() or
            np.iscomplex(eigvals).any() or
                np.iscomplex(eigvecs).any()):
            print(f'skip a spot with invalid eigen value(s): {eigvals}')
            continue
        # https://stackoverflow.com/questions/22146383/covariance-matrix-of-an-ellipse
        # https://github.com/scikit-image/scikit-image/blob/master/skimage/measure/_regionprops.py#L288
        radii = 2 * np.sqrt(eigvals)
        radii /= (c_ratio / region.mean_intensity)
        radii = np.array([radii[i] if 0 < radii[i] else max(
            r_min, scales[i]) for i in range(len(radii))])
        if (radii < r_min).any():
            # print(f'skip a spot with radii {radii} below threshold {r_min}')
            continue
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
        spots.append(spot)


def export_ctc_labels(config, spots_dict, redis_c=None):
    MIN_AREA_ELLIPSOID = 20
    timepoints = set([*range(config.t_start, config.t_end + 1, 1)])
    is_zip = config.savedir is None
    savedir = tempfile.mkdtemp() if is_zip else config.savedir
    for t, spots in spots_dict.items():
        if (redis_c is not None and
                int(redis_c.get(REDIS_KEY_STATE)) == TrainState.IDLE.value):
            print('aborted')
            if is_zip:
                shutil.rmtree(savedir)
            return False
        label = np.zeros(config.shape, np.uint16)
        for i in range(16):
            for spot in spots:
                centroid = np.array(spot['pos'][::-1])
                covariance = np.array(spot['covariance'][::-1]).reshape(3, 3)
                radii, rotation = np.linalg.eigh(covariance)
                radii = np.sqrt(radii)
                dd_outer, rr_outer, cc_outer = ellipsoid(
                    centroid,
                    radii * (1 - 0.05 * i),
                    rotation,
                    config.scales,
                    label.shape,
                    MIN_AREA_ELLIPSOID
                )
                label[dd_outer, rr_outer, cc_outer] = spot['value']
        # ensure that each spot is labeled at least its center voxcel
        for spot in spots:
            centroid = np.array(spot['pos'][::-1])
            dd_center = int(centroid[0] / config.scales[0])
            rr_center = int(centroid[1] / config.scales[1])
            cc_center = int(centroid[2] / config.scales[2])
            label[dd_center, rr_center, cc_center] = spot['value']
        skimage.io.imsave(
            os.path.join(savedir, f'mask{t:03d}.tif'),
            label[:, None],
            imagej=True,
            compress=6,
        )
        timepoints.discard(t)
    # Save blank images for unlabeled frames
    label = np.zeros(config.shape, np.uint16)
    for t in timepoints:
        skimage.io.imsave(
            os.path.join(savedir, f'mask{t:03d}.tif'),
            label[:, None],
            imagej=True,
            compress=6,
        )
    if is_zip:
        zip_path = shutil.make_archive('/tmp/archive.zip', 'zip', savedir)
        shutil.rmtree(savedir)
        return zip_path
    return True


def init_seg_models(config):
    models = [UNet.three_class_segmentation(
        config.keep_axials,
        device=config.device,
        is_3d=config.is_3d,
    ) for i in range(config.n_models)]
    n_dims = 2 + config.is_3d  # 3 or 2
    input_shape = zarr.open(config.zpath_input, mode='r').shape[-n_dims:]
    n_crops = config.n_crops
    train_dataset = AutoencoderDatasetZarr(
        config.zpath_input,
        input_shape,
        config.crop_size,
        n_crops,
        scales=config.scales,
        scale_factor_base=0.2,
    )
    loss = AutoencoderLoss()
    loss = loss.to(config.device)
    train_loader = du.DataLoader(
        train_dataset, shuffle=True, batch_size=1)
    for i, lr in enumerate([1e-2, 1e-3, 1e-4]):
        optimizers = [torch.optim.Adam(
            model.parameters(), lr=lr) for model in models]
        for model, optimizer in zip(models, optimizers):
            train(model,
                  config.device,
                  train_loader,
                  optimizer=optimizer,
                  loss_fn=loss,
                  epoch=i,
                  log_interval=1)
    torch.save(models[0].state_dict() if len(models) == 1 else [
        model.state_dict() for model in models], config.model_path)
    return models


def init_flow_models(config):
    models = [FlowResNet.three_dimensional_flow(
        config.keep_axials,
        device=config.device,
        is_3d=config.is_3d,
    ) for i in range(config.n_models)]
    torch.save(models[0].state_dict() if len(models) == 1 else [
        model.state_dict() for model in models], config.model_path)
    return models
