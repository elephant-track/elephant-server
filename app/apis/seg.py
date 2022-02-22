# Copyright (c) 2022, Ko Sugawara
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
import collections
import json
import gc
import time
from pathlib import Path

from celery import shared_task
from filelock import FileLock
from flask import jsonify
from flask import make_response
from flask import request
from flask_restx import Namespace
from flask_restx import Resource
import numpy as np
from tensorflow.data import TFRecordDataset
from tensorflow.core.util import event_pb2
import torch
import torch.multiprocessing as mp
import zarr

from elephant.common import detect_spots
from elephant.common import init_seg_models
from elephant.common import run_train_seg
from elephant.config import SegmentationEvalConfig
from elephant.config import SegmentationTrainConfig
from elephant.config import ResetConfig
from elephant.datasets import get_input_at
from elephant.logging import logger
from elephant.logging import publish_mq
from elephant.redis_util import get_state
from elephant.redis_util import redis_client
from elephant.redis_util import TrainState
from elephant.redis_util import REDIS_KEY_STATE
from elephant.redis_util import REDIS_KEY_TIMEPOINT
from elephant.redis_util import REDIS_KEY_UPDATE_ONGOING_SEG
from elephant.util import get_device
from elephant.util import to_fancy_index
from elephant.util.ellipse import ellipse
from elephant.util.ellipsoid import ellipsoid

api = Namespace('seg', description='Seg APIs')


@shared_task()
def train_seg_task(spot_indices, batch_size, crop_size, class_weights,
                   false_weight, model_path, n_epochs, keep_axials, scales,
                   lr, n_crops, is_3d, is_livemode, scale_factor_base,
                   rotation_angle, contrast, zpath_input, zpath_seg_label,
                   log_interval, log_dir, step_offset=0, epoch_start=0,
                   is_cpu=False, is_mixed_precision=True, cache_maxbytes=None,
                   memmap_dir=None):
    if not torch.cuda.is_available():
        is_cpu = True
    world_size = 2 if is_cpu else torch.cuda.device_count()
    if 1 < world_size:
        mp.spawn(run_train_seg,
                 args=(world_size, spot_indices, batch_size, crop_size,
                       class_weights, false_weight, model_path, n_epochs,
                       keep_axials, scales, lr, n_crops, is_3d, is_livemode,
                       scale_factor_base, rotation_angle, contrast, zpath_input,
                       zpath_seg_label, log_interval, log_dir, step_offset,
                       epoch_start, is_cpu, is_mixed_precision, cache_maxbytes,
                       memmap_dir),
                 nprocs=world_size,
                 join=True)
    else:
        run_train_seg(torch.device('cpu') if is_cpu else torch.device('cuda'),
                      world_size, spot_indices, batch_size, crop_size,
                      class_weights, false_weight, model_path, n_epochs,
                      keep_axials, scales, lr, n_crops, is_3d, is_livemode,
                      scale_factor_base, rotation_angle, contrast, zpath_input,
                      zpath_seg_label, log_interval, log_dir, step_offset,
                      epoch_start, is_cpu, is_mixed_precision, cache_maxbytes,
                      memmap_dir)


@shared_task()
def detect_spots_task(device, model_path, keep_axials=(True,) * 4, is_pad=False,
                      is_3d=True, crop_size=(16, 384, 384), scales=None,
                      cache_maxbytes=None, use_2d=False, use_median=False,
                      patch_size=None, crop_box=None, c_ratio=0.4, p_thresh=0.5,
                      r_min=0, r_max=1e6, output_prediction=False,
                      zpath_input=None, zpath_seg_output=None, timepoint=None,
                      tiff_input=None, memmap_dir=None, batch_size=1):
    """
    Detect spots at the specified timepoint.

    Parameters
    ----------

    Returns
    -------
    spots : list
        Detected spots as list. None is returned on error or cancel.
    """
    return detect_spots(device, model_path, keep_axials, is_pad, is_3d, crop_size,
                        scales, cache_maxbytes, use_2d, use_median, patch_size,
                        crop_box, c_ratio, p_thresh, r_min, r_max,
                        output_prediction, zpath_input, zpath_seg_output,
                        timepoint, tiff_input, memmap_dir, batch_size)


def _dilate_2d_indices(rr, cc, shape):
    if len(rr) != len(cc):
        raise RuntimeError('indices should have the same length')
    n_pixels = len(rr)
    rr_dilate = np.array([0, ] * (n_pixels * 3 ** 2))
    cc_dilate = np.copy(rr_dilate)
    offset = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            rr_dilate[
                offset:offset + n_pixels
            ] = (rr + dy).clip(0, shape[0] - 1)
            cc_dilate[
                offset:offset + n_pixels
            ] = (cc + dx).clip(0, shape[1] - 1)
            offset += n_pixels
    unique_dilate = np.unique(
        np.stack((rr_dilate, cc_dilate)), axis=1)
    return unique_dilate[0], unique_dilate[1]


def _dilate_3d_indices(dd, rr, cc, shape):
    if len(dd) != len(rr) or len(dd) != len(cc):
        raise RuntimeError('indices should have the same length')
    n_pixels = len(dd)
    dd_dilate = np.array([0, ] * (n_pixels * 3 ** 3))
    rr_dilate = np.copy(dd_dilate)
    cc_dilate = np.copy(dd_dilate)
    offset = 0
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                dd_dilate[
                    offset:offset + n_pixels
                ] = (dd + dz).clip(0, shape[0] - 1)
                rr_dilate[
                    offset:offset + n_pixels
                ] = (rr + dy).clip(0, shape[1] - 1)
                cc_dilate[
                    offset:offset + n_pixels
                ] = (cc + dx).clip(0, shape[2] - 1)
                offset += n_pixels
    unique_dilate = np.unique(
        np.stack((dd_dilate, rr_dilate, cc_dilate)), axis=1)
    return unique_dilate[0], unique_dilate[1], unique_dilate[2]


def _update_seg_labels(spots_dict, scales, zpath_input, zpath_seg_label,
                       zpath_seg_label_vis, auto_bg_thresh=0, c_ratio=0.5,
                       is_livemode=False, memmap_dir=None):
    if is_livemode:
        assert len(spots_dict.keys()) == 1
    za_input = zarr.open(zpath_input, mode='r')
    za_label = zarr.open(zpath_seg_label, mode='a')
    za_label_vis = zarr.open(zpath_seg_label_vis, mode='a')
    keyorder = ['tp', 'fp', 'tn', 'fn', 'tb', 'fb']
    MIN_AREA_ELLIPSOID = 9
    img_shape = za_input.shape[1:]
    n_dims = len(img_shape)
    keybase = Path(za_label.store.path).parent.name
    for t, spots in spots_dict.items():
        label_indices = set()
        if 0 < auto_bg_thresh:
            label = np.where(
                get_input_at(
                    za_input, t, memmap_dir=memmap_dir
                ) < auto_bg_thresh,
                1,
                0
            ).astype('uint8')
            label_indices.update(
                tuple(map(tuple, np.stack(np.nonzero(label), axis=1).tolist()))
            )
        else:
            label = np.zeros(img_shape, dtype='uint8')
        label_vis = np.zeros(img_shape + (3,), dtype='uint8')
        cnt = collections.Counter({x: 0 for x in keyorder})
        for spot in spots:
            if get_state() == TrainState.IDLE.value:
                logger().info('update aborted')
                raise KeyboardInterrupt
            cnt[spot['tag']] += 1
            centroid = np.array(spot['pos'][::-1])
            centroid = centroid[-n_dims:]
            covariance = np.array(spot['covariance'][::-1]).reshape(3, 3)
            covariance = covariance[-n_dims:, -n_dims:]
            radii, rotation = np.linalg.eigh(covariance)
            radii = np.sqrt(radii)
            if n_dims == 3:
                draw_func = ellipsoid
                dilate_func = _dilate_3d_indices
            else:
                draw_func = ellipse
                dilate_func = _dilate_2d_indices
            indices_outer = draw_func(
                centroid,
                radii,
                rotation,
                scales,
                img_shape,
                MIN_AREA_ELLIPSOID
            )
            label_indices.update(
                tuple(map(tuple, np.stack(indices_outer, axis=1).tolist()))
            )
            if spot['tag'] in ['tp', 'tb', 'tn']:
                label_offset = 0
                label_vis_value = 255
            else:
                label_offset = 3
                label_vis_value = 127
            cond_outer_1 = np.fmod(label[indices_outer] - 1, 3) <= 1
            if spot['tag'] in ('tp', 'fn'):
                indices_inner = draw_func(
                    centroid,
                    radii * c_ratio,
                    rotation,
                    scales,
                    img_shape,
                    MIN_AREA_ELLIPSOID
                )
                indices_inner_p = dilate_func(*indices_inner, img_shape)
                label[indices_outer] = np.where(
                    cond_outer_1,
                    2 + label_offset,
                    label[indices_outer]
                )
                label_vis[to_fancy_index(*indices_outer, 1)] = np.where(
                    cond_outer_1,
                    label_vis_value,
                    label_vis[to_fancy_index(*indices_outer, 1)]
                )
                label[indices_inner_p] = 2 + label_offset
                label_vis[
                    to_fancy_index(*indices_inner_p, 1)] = label_vis_value
                cond_inner = np.fmod(label[indices_inner] - 1, 3) <= 2
                label[indices_inner] = np.where(
                    cond_inner,
                    3 + label_offset,
                    label[indices_inner]
                )
                label_vis[to_fancy_index(*indices_inner, 2)] = np.where(
                    cond_inner,
                    label_vis_value,
                    label_vis[to_fancy_index(*indices_inner, 2)]
                )
            elif spot['tag'] in ('tb', 'fb'):
                label[indices_outer] = np.where(
                    cond_outer_1,
                    2 + label_offset,
                    label[indices_outer]
                )
                label_vis[to_fancy_index(*indices_outer, 1)] = np.where(
                    cond_outer_1,
                    label_vis_value,
                    label_vis[to_fancy_index(*indices_outer, 1)]
                )
            elif spot['tag'] in ('tn', 'fp'):
                cond_outer_0 = np.fmod(label[indices_outer] - 1, 3) <= 0
                label[indices_outer] = np.where(
                    cond_outer_0,
                    1 + label_offset,
                    label[indices_outer]
                )
                label_vis[to_fancy_index(*indices_outer, 0)] = np.where(
                    cond_outer_0,
                    label_vis_value,
                    label_vis[to_fancy_index(*indices_outer, 0)]
                )
        logger().info('frame:{}, {}'.format(
            t, sorted(cnt.items(), key=lambda i: keyorder.index(i[0]))))
        target = tuple(np.array(list(label_indices)).T)
        target_t = to_fancy_index(t, *target)
        target_vis = tuple(
            np.column_stack([to_fancy_index(*target, c) for c in range(3)])
        )
        target_vis_t = to_fancy_index(t, *target_vis)
        for chunk in Path(za_label.store.path).glob(f'{t}.*'):
            chunk.unlink()
        for chunk in Path(za_label_vis.store.path).glob(f'{t}.*'):
            chunk.unlink()
        za_label[target_t] = label[target]
        za_label_vis[target_vis_t] = label_vis[target_vis]
        za_label.attrs[f'label.indices.{t}'] = list(label_indices)
        za_label.attrs['updated'] = True
        if memmap_dir:
            fpath = Path(memmap_dir) / f'{keybase}-t{t}-seglabel.dat'
            lock = FileLock(str(fpath) + '.lock')
            with lock:
                if fpath.exists():
                    logger().info(f'remove {fpath}')
                    fpath.unlink()
        if is_livemode:
            if redis_client.get(REDIS_KEY_TIMEPOINT):
                msg = 'Last update/training is ongoing'
                logger().error(msg)
                return make_response(jsonify(error=msg), 400)
            redis_client.set(REDIS_KEY_TIMEPOINT, t)
    return make_response(jsonify({'completed': True}))


@ api.route('/predict')
class Predict(Resource):
    @ api.doc()
    def post(self):
        '''
        Predict seg.

        '''
        if request.headers['Content-Type'] != 'application/json':
            msg = 'Content-Type should be application/json'
            return make_response(jsonify(error=msg), 400)
        state = get_state()
        while (state == TrainState.WAIT.value):
            logger().info(f'waiting @{request.path}')
            time.sleep(1)
            state = get_state()
            if (state == TrainState.IDLE.value):
                return make_response(jsonify({'completed': False}))
        try:
            req_json = request.get_json()
            req_json['device'] = get_device()
            config = SegmentationEvalConfig(req_json)
            logger().info(config)
            redis_client.set(REDIS_KEY_STATE, TrainState.WAIT.value)
            async_result = detect_spots_task.delay(
                str(config.device), config.model_path, config.keep_axials,
                config.is_pad, config.is_3d, config.crop_size, config.scales,
                config.cache_maxbytes, config.use_2d, config.use_median,
                config.patch_size, config.crop_box, config.c_ratio,
                config.p_thresh, config.r_min, config.r_max,
                config.output_prediction, config.zpath_input,
                config.zpath_seg_output, config.timepoint, None,
                config.memmap_dir, config.batch_size,
            )
            while not async_result.ready():
                if (redis_client is not None and
                    get_state()
                        == TrainState.IDLE.value):
                    logger().info('prediction aborted')
                    return make_response(
                        jsonify({'spots': [], 'completed': False})
                    )
            if async_result.failed():
                raise async_result.result
            spots = async_result.result
            if spots is None:
                logger().info('prediction aborted')
                return make_response(jsonify({'spots': [], 'completed': False}))
            publish_mq('prediction', 'Prediction updated')
        except Exception as e:
            logger().exception('Failed in detect_spots')
            return make_response(jsonify(error=f'Exception: {e}'), 500)
        finally:
            gc.collect()
            torch.cuda.empty_cache()
            if get_state() != TrainState.IDLE.value:
                redis_client.set(REDIS_KEY_STATE, state)
        return make_response(jsonify({'spots': spots, 'completed': True}))


@ api.route('/update')
class Update(Resource):
    @ api.doc()
    def post(self):
        '''
        Update seg label.

        '''
        if request.headers['Content-Type'] != 'application/json':
            msg = 'Content-Type should be application/json'
            logger().error(msg)
            return make_response(jsonify(error=msg), 400)
        state = get_state()
        while (state == TrainState.WAIT.value):
            logger().info(f'waiting @{request.path}')
            time.sleep(1)
            state = get_state()
            if (state == TrainState.IDLE.value):
                return make_response(jsonify({'completed': False}))
        if redis_client.get(REDIS_KEY_UPDATE_ONGOING_SEG):
            msg = 'Last update is ongoing'
            logger().error(msg)
            return make_response(jsonify(error=msg), 400)
        try:
            redis_client.set(REDIS_KEY_UPDATE_ONGOING_SEG, 1)
            redis_client.set(REDIS_KEY_STATE, TrainState.WAIT.value)
            req_json = request.get_json()
            req_json['device'] = get_device()
            config = SegmentationTrainConfig(req_json)
            logger().info(config)
            if config.is_livemode and redis_client.get(REDIS_KEY_TIMEPOINT):
                msg = 'Last update/training is ongoing'
                logger().error(msg)
                return make_response(jsonify(error=msg), 400)
            if req_json.get('reset'):
                try:
                    zarr.open_like(
                        zarr.open(config.zpath_seg_label, mode='r'),
                        config.zpath_seg_label,
                        mode='w'
                    )
                    zarr.open_like(
                        zarr.open(config.zpath_seg_label_vis, mode='r'),
                        config.zpath_seg_label_vis,
                        mode='w'
                    )
                except RuntimeError as e:
                    logger().exception('Failed in opening zarr')
                    return make_response(
                        jsonify(error=f'Runtime Error: {e}'),
                        500
                    )
                except Exception as e:
                    logger().exception('Failed in opening zarr')
                    return make_response(jsonify(error=f'Exception: {e}'), 500)
                return make_response(jsonify({'completed': True}))

            spots_dict = collections.defaultdict(list)
            for spot in req_json.get('spots'):
                spots_dict[spot['t']].append(spot)
            if not spots_dict:
                msg = 'nothing to update'
                logger().error(msg)
                return make_response(jsonify(error=msg), 400)
            if config.is_livemode and len(spots_dict.keys()) != 1:
                msg = 'Livemode should update only a single timepoint'
                logger().error(msg)
                return make_response(jsonify(error=msg), 400)
            spots_dict = collections.OrderedDict(sorted(spots_dict.items()))

            try:
                response = _update_seg_labels(spots_dict,
                                              config.scales,
                                              config.zpath_input,
                                              config.zpath_seg_label,
                                              config.zpath_seg_label_vis,
                                              config.auto_bg_thresh,
                                              config.c_ratio,
                                              config.is_livemode,
                                              memmap_dir=config.memmap_dir)
            except KeyboardInterrupt:
                return make_response(jsonify({'completed': False}))
            except Exception as e:
                logger().exception('Failed in _update_seg_labels')
                return make_response(jsonify(error=f'Exception: {e}'), 500)
        finally:
            if get_state() != TrainState.IDLE.value:
                redis_client.set(REDIS_KEY_STATE, state)
            redis_client.delete(REDIS_KEY_UPDATE_ONGOING_SEG)
        return response


@ api.route('/train')
class Train(Resource):
    @ api.doc()
    def post(self):
        '''
        Train seg model.

        '''
        if request.headers['Content-Type'] != 'application/json':
            msg = 'Content-Type should be application/json'
            logger().error(msg)
            return make_response(jsonify(error=msg), 400)
        try:
            req_json = request.get_json()
            req_json['device'] = get_device()
            config = SegmentationTrainConfig(req_json)
            logger().info(config)
            if config.n_crops < 1:
                msg = 'n_crops should be a positive number'
                logger().error(msg)
                return make_response(jsonify(error=msg), 400)

            spots_dict = collections.defaultdict(list)
            for spot in req_json.get('spots'):
                spots_dict[spot['t']].append(spot)
            if not (spots_dict or config.is_livemode):
                msg = 'nothing to train'
                logger().error(msg)
                return make_response(jsonify(error=msg), 400)
            spots_dict = collections.OrderedDict(sorted(spots_dict.items()))

            if get_state() != TrainState.IDLE.value:
                msg = 'Process is running'
                logger().error(msg)
                return make_response(jsonify(error=msg), 500)
            redis_client.set(REDIS_KEY_STATE, TrainState.RUN.value)
            if config.is_livemode:
                redis_client.delete(REDIS_KEY_TIMEPOINT)
            else:
                try:
                    _update_seg_labels(spots_dict,
                                       config.scales,
                                       config.zpath_input,
                                       config.zpath_seg_label,
                                       config.zpath_seg_label_vis,
                                       config.auto_bg_thresh,
                                       config.c_ratio,
                                       memmap_dir=config.memmap_dir)
                except KeyboardInterrupt:
                    return make_response(jsonify({'completed': False}))
            step_offset = 0
            for path in sorted(Path(config.log_dir).glob('event*')):
                try:
                    *_, last_record = TFRecordDataset(str(path))
                    last = event_pb2.Event.FromString(last_record.numpy()).step
                    step_offset = max(step_offset, last+1)
                except Exception:
                    pass
            epoch_start = 0
            async_result = train_seg_task.delay(
                list(spots_dict.keys()), config.batch_size,
                config.crop_size, config.class_weights,
                config.false_weight, config.model_path, config.n_epochs,
                config.keep_axials, config.scales, config.lr,
                config.n_crops, config.is_3d, config.is_livemode,
                config.scale_factor_base, config.rotation_angle,
                config.contrast, config.zpath_input, config.zpath_seg_label,
                config.log_interval, config.log_dir, step_offset, epoch_start,
                config.is_cpu(), config.is_mixed_precision,
                config.cache_maxbytes, config.memmap_dir,
            )
            while not async_result.ready():
                if (redis_client is not None and
                        get_state()
                        == TrainState.IDLE.value):
                    logger().info('training aborted')
                    return make_response(jsonify({'completed': False}))
        except Exception as e:
            logger().exception('Failed in train_seg')
            return make_response(jsonify(error=f'Exception: {e}'), 500)
        finally:
            torch.cuda.empty_cache()
            redis_client.set(REDIS_KEY_STATE, TrainState.IDLE.value)
        return make_response(jsonify({'completed': True}))


@ api.route('/reset')
class Reset(Resource):
    @ api.doc()
    def post(self):
        '''
        Reset seg model.

        '''
        if all(ctype not in request.headers['Content-Type'] for ctype
               in ('multipart/form-data', 'application/json')):
            msg = ('Content-Type should be multipart/form-data or '
                   'application/json')
            logger().error(msg)
            return make_response(jsonify(error=msg), 400)
        if get_state() != TrainState.IDLE.value:
            msg = 'Process is running. Model cannot be reset.'
            logger().error(msg)
            return make_response(jsonify(error=msg), 500)
        try:
            if 'multipart/form-data' in request.headers['Content-Type']:
                print(request.form)
                req_json = json.loads(request.form.get('data'))
                file = request.files['file']
                checkpoint = torch.load(file.stream)
                state_dicts = checkpoint if isinstance(
                    checkpoint, list) else [checkpoint]
                req_json['url'] = None
            else:
                req_json = request.get_json()
                state_dicts = None
            req_json['device'] = get_device()
            config = ResetConfig(req_json)
            logger().info(config)
            redis_client.set(REDIS_KEY_STATE, TrainState.RUN.value)
            init_seg_models(config.model_path,
                            config.keep_axials,
                            config.device,
                            config.is_3d,
                            config.n_models,
                            config.n_crops,
                            config.zpath_input,
                            config.crop_size,
                            config.scales,
                            url=config.url,
                            state_dicts=state_dicts,
                            is_cpu=config.is_cpu(),
                            cache_maxbytes=config.cache_maxbytes)
        except RuntimeError as e:
            logger().exception('Failed in init_seg_models')
            return make_response(jsonify(error=f'Runtime Error: {e}'), 500)
        except Exception as e:
            logger().exception('Failed in init_seg_models')
            return make_response(jsonify(error=f'Exception: {e}'), 500)
        finally:
            gc.collect()
            torch.cuda.empty_cache()
            redis_client.set(REDIS_KEY_STATE, TrainState.IDLE.value)
        return make_response(jsonify({'completed': True}))
