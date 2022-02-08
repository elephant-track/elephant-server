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
"""Flask endpoints."""

import collections
import gc
import json
import os
from pathlib import Path
import tempfile
import weakref

from celery import Celery
from filelock import FileLock
from flask import Flask
from flask import g
from flask import jsonify
from flask import make_response
from flask import request
from flask import send_file
from flask_redis import FlaskRedis
import numpy as np
import nvsmi
import psutil
from tensorflow.data import TFRecordDataset
from tensorflow.core.util import event_pb2
import time
import torch
import torch.multiprocessing as mp
import werkzeug
import zarr

from elephant.common import detect_spots
from elephant.common import export_ctc_labels
from elephant.common import init_flow_models
from elephant.common import init_seg_models
from elephant.common import spots_with_flow
from elephant.common import run_train_seg
from elephant.common import run_train_flow
from elephant.config import DATASETS_DIR
from elephant.config import BaseConfig
from elephant.config import ExportConfig
from elephant.config import FlowEvalConfig
from elephant.config import FlowTrainConfig
from elephant.config import ResetConfig
from elephant.config import SegmentationEvalConfig
from elephant.config import SegmentationTrainConfig
from elephant.datasets import get_input_at
from elephant.logging import logger
from elephant.logging import publish_mq
from elephant.redis_util import REDIS_KEY_LR
from elephant.redis_util import REDIS_KEY_NCROPS
from elephant.redis_util import REDIS_KEY_STATE
from elephant.redis_util import REDIS_KEY_TIMEPOINT
from elephant.redis_util import REDIS_KEY_UPDATE_ONGOING_FLOW
from elephant.redis_util import REDIS_KEY_UPDATE_ONGOING_SEG
from elephant.redis_util import TrainState
from elephant.tool import dataset as dstool
from elephant.util import get_device
from elephant.util import to_fancy_index
from elephant.util.ellipse import ellipse
from elephant.util.ellipsoid import ellipsoid


def make_celery(app):
    celery = Celery(
        app.import_name,
        backend=app.config['result_backend'],
        broker=app.config['broker_url']
    )
    celery.conf.update(app.config)

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery


app = Flask(__name__)
app.config.update(
    broker_url='redis://localhost:6379',
    result_backend='redis://localhost:6379',
    worker_redirect_stdouts=False,
)

celery = make_celery(app)
redis_client = FlaskRedis(app)
redis_client.set(REDIS_KEY_STATE, TrainState.IDLE.value)


@celery.task()
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


@ celery.task()
def train_flow_task(spot_indices, batch_size, crop_size, model_path, n_epochs,
                    keep_axials, scales, lr, n_crops, is_3d, scale_factor_base,
                    rotation_angle, zpath_input, zpath_flow_label,
                    log_interval, log_dir, step_offset=0, epoch_start=0,
                    is_cpu=False, is_mixed_precision=True, cache_maxbytes=None,
                    memmap_dir=None):
    if not torch.cuda.is_available():
        is_cpu = True
    world_size = 2 if is_cpu else torch.cuda.device_count()
    if 1 < world_size:
        mp.spawn(run_train_flow,
                 args=(world_size, spot_indices, batch_size, crop_size,
                       model_path, n_epochs, keep_axials, scales, lr,
                       n_crops, is_3d, scale_factor_base, rotation_angle,
                       zpath_input, zpath_flow_label, log_interval, log_dir,
                       step_offset, epoch_start, is_cpu, is_mixed_precision,
                       cache_maxbytes, memmap_dir),
                 nprocs=world_size,
                 join=True)
    else:
        run_train_flow(torch.device('cpu') if is_cpu else torch.device('cuda'),
                       world_size, spot_indices, batch_size, crop_size,
                       model_path, n_epochs, keep_axials, scales, lr,
                       n_crops, is_3d, scale_factor_base, rotation_angle,
                       zpath_input, zpath_flow_label, log_interval, log_dir,
                       step_offset, epoch_start, is_cpu, is_mixed_precision,
                       cache_maxbytes, memmap_dir)


@ celery.task()
def detect_spots_task(device, model_path, keep_axials=(True,) * 4, is_3d=True,
                      crop_size=(16, 384, 384), scales=None,
                      cache_maxbytes=None, use_2d=False, use_median=False,
                      patch_size=None, crop_box=None, c_ratio=0.4, p_thresh=0.5,
                      r_min=0, r_max=1e6, output_prediction=False,
                      zpath_input=None, zpath_seg_output=None, timepoint=None,
                      tiff_input=None, memmap_dir=None, batch_size=1):
    return detect_spots(device, model_path, keep_axials, is_3d, crop_size,
                        scales, cache_maxbytes, use_2d, use_median, patch_size,
                        crop_box, c_ratio, p_thresh, r_min, r_max,
                        output_prediction, zpath_input, zpath_seg_output,
                        timepoint, tiff_input, memmap_dir, batch_size)


class FileRemover(object):
    '''
    https: // stackoverflow.com/a/32132035
    '''

    def __init__(self):
        self.weak_references = dict()  # weak_ref -> filepath to remove

    def cleanup_once_done(self, response, filepath):
        wr = weakref.ref(response, self._do_cleanup)
        self.weak_references[wr] = filepath

    def _do_cleanup(self, wr):
        filepath = self.weak_references[wr]
        print('Deleting %s' % filepath)
        os.remove(filepath)


file_remover = FileRemover()


def device():
    if 'device' not in g:
        g.device = get_device()
    return g.device


@ app.before_request
def log_before_request():
    if request.endpoint != 'get_gpus':
        logger().info(f'START {request.method} {request.path}')


@ app.after_request
def log_after_request(response):
    if request.endpoint != 'get_gpus':
        logger().info(
            f'DONE {request.method} {request.path} => [{response.status}]'
        )
    return response


@ app.route('/state', methods=['GET', 'POST'])
def state():
    if request.method == 'POST':
        if request.headers['Content-Type'] != 'application/json':
            msg = 'Content-Type should be application/json'
            logger().error(msg)
            return jsonify(error=msg), 400
        req_json = request.get_json()
        state = req_json.get(REDIS_KEY_STATE)
        if (not isinstance(state, int) or
                state not in TrainState._value2member_map_):
            msg = f'Invalid state: {state}'
            logger().error(msg)
            return jsonify(res=msg), 400
        redis_client.set(REDIS_KEY_STATE, state)
    return jsonify(success=True, state=int(redis_client.get(REDIS_KEY_STATE)))


@ app.route('/params', methods=['GET', 'POST'])
def params():
    if request.method == 'POST':
        if request.headers['Content-Type'] != 'application/json':
            msg = 'Content-Type should be application/json'
            logger().error(msg)
            return jsonify(error=msg), 400
        req_json = request.get_json()
        lr = req_json.get('lr')
        if not isinstance(lr, float) or lr < 0:
            msg = f'Invalid learning rate: {lr}'
            logger().error(msg)
            return jsonify(error=msg), 400
        n_crops = req_json.get('n_crops')
        if not isinstance(n_crops, int) or n_crops < 0:
            msg = f'Invalid number of crops: {n_crops}'
            logger().error(msg)
            return jsonify(error=msg), 400
        redis_client.set(REDIS_KEY_LR, str(lr))
        redis_client.set(REDIS_KEY_NCROPS, str(n_crops))
        logger().info(f'[params updated] lr: {lr}, n_crops: {n_crops}')
    return jsonify(success=True,
                   lr=float(redis_client.get(REDIS_KEY_LR)),
                   n_crops=int(redis_client.get(REDIS_KEY_NCROPS)))


def _update_flow_labels(spots_dict,
                        scales,
                        zpath_flow_label,
                        flow_norm_factor):
    za_label = zarr.open(zpath_flow_label, mode='a')
    MIN_AREA_ELLIPSOID = 9
    n_dims = len(za_label.shape) - 2
    for t, spots in spots_dict.items():
        label = np.zeros(za_label.shape[1:], dtype='float32')
        label_indices = set()
        for spot in spots:
            if int(redis_client.get(REDIS_KEY_STATE)) == TrainState.IDLE.value:
                logger().info('update aborted')
                return jsonify({'completed': False})
            centroid = np.array(spot['pos'][::-1])
            centroid = centroid[-n_dims:]
            covariance = np.array(spot['covariance'][::-1]).reshape(3, 3)
            covariance = covariance[-n_dims:, -n_dims:]
            radii, rotation = np.linalg.eigh(covariance)
            radii = np.sqrt(radii)
            draw_func = ellipsoid if n_dims == 3 else ellipse
            indices = draw_func(
                centroid,
                radii,
                rotation,
                scales,
                label.shape[-n_dims:],
                MIN_AREA_ELLIPSOID
            )
            weight = 1  # if spot['tag'] in ['tp'] else false_weight
            displacement = spot['displacement']  # X, Y, Z
            for i in range(n_dims):
                label[i][indices] = (
                    displacement[i] / scales[-1 - i] / flow_norm_factor[i])
            label[-1][indices] = weight  # last channels is for weight
            label_indices.update(
                tuple(map(tuple, np.stack(indices, axis=1).tolist())))
        logger().info(f'frame:{t+1}, {len(spots)} linkings')
        za_label.attrs[f'label.indices.{t}'] = list(label_indices)
        za_label.attrs['updated'] = True
        za_label[t] = label
    return jsonify({'completed': True})


@app.route('/update/flow', methods=['POST'])
def update_flow_labels():
    if request.headers['Content-Type'] != 'application/json':
        msg = 'Content-Type should be application/json'
        logger().error(msg)
        return jsonify(error=msg), 400
    state = int(redis_client.get(REDIS_KEY_STATE))
    while (state == TrainState.WAIT.value):
        logger().info(f'waiting @{request.path}')
        time.sleep(1)
        state = int(redis_client.get(REDIS_KEY_STATE))
        if (state == TrainState.IDLE.value):
            return jsonify({'completed': False})
    if redis_client.get(REDIS_KEY_UPDATE_ONGOING_FLOW):
        msg = 'Last update is ongoing'
        logger().error(msg)
        return jsonify(error=msg), 400
    try:
        redis_client.set(REDIS_KEY_STATE, TrainState.WAIT.value)
        req_json = request.get_json()
        req_json['device'] = device()
        config = FlowTrainConfig(req_json)
        logger().info(config)
        if req_json.get('reset'):
            zarr.open_like(zarr.open(config.zpath_flow_label, mode='r'),
                           config.zpath_flow_label,
                           mode='w')
            return jsonify({'completed': True})

        spots_dict = collections.defaultdict(list)
        for spot in req_json.get('spots'):
            spots_dict[spot['t']].append(spot)
        if not spots_dict:
            msg = 'nothing to update'
            logger().error(msg)
            return jsonify(error=msg), 400
        spots_dict = collections.OrderedDict(sorted(spots_dict.items()))

        response = _update_flow_labels(spots_dict,
                                       config.scales,
                                       config.zpath_flow_label,
                                       config.flow_norm_factor)
    except RuntimeError as e:
        logger().exception('Failed in update_flow_labels')
        return jsonify(error=f'Runtime Error: {e}'), 500
    except Exception as e:
        logger().exception('Failed in update_flow_labels')
        return jsonify(error=f'Exception: {e}'), 500
    finally:
        if int(redis_client.get(REDIS_KEY_STATE)) != TrainState.IDLE.value:
            redis_client.set(REDIS_KEY_STATE, state)
        redis_client.delete(REDIS_KEY_UPDATE_ONGOING_FLOW)
    return response


@app.route('/train/flow', methods=['POST'])
def train_flow():
    if request.headers['Content-Type'] != 'application/json':
        msg = 'Content-Type should be application/json'
        logger().error(msg)
        return jsonify(error=msg), 400
    try:
        req_json = request.get_json()
        req_json['device'] = device()
        config = FlowTrainConfig(req_json)
        logger().info(config)
        if config.n_crops < 1:
            msg = 'n_crops should be a positive number'
            logger().error(msg)
            return jsonify(error=msg), 400

        spots_dict = collections.defaultdict(list)
        for spot in req_json.get('spots'):
            spots_dict[spot['t']].append(spot)
        if not spots_dict:
            msg = 'nothing to train'
            logger().error(msg)
            return jsonify(error=msg), 400
        spots_dict = collections.OrderedDict(sorted(spots_dict.items()))

        if int(redis_client.get(REDIS_KEY_STATE)) != TrainState.IDLE.value:
            msg = 'Process is running'
            logger().error(msg)
            return jsonify(error='Process is running'), 500
        redis_client.set(REDIS_KEY_STATE, TrainState.RUN.value)
        _update_flow_labels(spots_dict,
                            config.scales,
                            config.zpath_flow_label,
                            config.flow_norm_factor)
        step_offset = 0
        for path in sorted(Path(config.log_dir).glob('event*')):
            try:
                *_, last_record = TFRecordDataset(str(path))
                last = event_pb2.Event.FromString(last_record.numpy()).step
                step_offset = max(step_offset, last+1)
            except Exception:
                pass
        epoch_start = 0
        train_flow_task.delay(
            list(spots_dict.keys()), config.batch_size, config.crop_size,
            config.model_path, config.n_epochs, config.keep_axials,
            config.scales, config.lr, config.n_crops, config.is_3d,
            config.scale_factor_base, config.rotation_angle,
            config.zpath_input, config.zpath_flow_label,
            config.log_interval, config.log_dir, step_offset, epoch_start,
            config.is_cpu(), config.is_mixed_precision, config.cache_maxbytes,
            config.memmap_dir,
        ).wait()
        if (redis_client is not None and
                int(redis_client.get(REDIS_KEY_STATE))
                == TrainState.IDLE.value):
            logger().info('training aborted')
            return jsonify({'completed': False})
    except KeyboardInterrupt:
        logger().info('training aborted')
        return jsonify({'completed': False})
    except RuntimeError as e:
        logger().exception('Failed in train_flow')
        return jsonify(error=f'Runtime Error: {e}'), 500
    except Exception as e:
        logger().exception('Failed in train_flow')
        return jsonify(error=f'Exception: {e}'), 500
    finally:
        gc.collect()
        torch.cuda.empty_cache()
        redis_client.set(REDIS_KEY_STATE, TrainState.IDLE.value)
    return jsonify({'completed': True})


@app.route('/predict/flow', methods=['POST'])
def predict_flow():
    if request.headers['Content-Type'] != 'application/json':
        msg = 'Content-Type should be application/json'
        logger().error(msg)
        return jsonify(error=msg), 400
    state = int(redis_client.get(REDIS_KEY_STATE))
    while (state == TrainState.WAIT.value):
        logger().info(f'waiting @{request.path}')
        time.sleep(1)
        state = int(redis_client.get(REDIS_KEY_STATE))
        if (state == TrainState.IDLE.value):
            logger().info(f'{request.path} cancelled')
            return jsonify({'completed': False})
    try:
        req_json = request.get_json()
        req_json['device'] = device()
        config = FlowEvalConfig(req_json)
        logger().info(config)
        redis_client.set(REDIS_KEY_STATE, TrainState.WAIT.value)
        spots = req_json.get('spots')
        res_spots = spots_with_flow(config, spots)
        if (redis_client is not None and
                int(redis_client.get(REDIS_KEY_STATE))
                == TrainState.IDLE.value):
            logger().info('prediction aborted')
            return jsonify({'spots': [], 'completed': False})
    except KeyboardInterrupt:
        logger().info('prediction aborted')
        return jsonify({'spots': [], 'completed': False})
    except RuntimeError as e:
        logger().exception('Failed in predict_flow')
        return jsonify(error=f'Runtime Error: {e}'), 500
    except Exception as e:
        logger().exception('Failed in predict_flow')
        return jsonify(error=f'Exception: {e}'), 500
    finally:
        gc.collect()
        torch.cuda.empty_cache()
        if int(redis_client.get(REDIS_KEY_STATE)) != TrainState.IDLE.value:
            redis_client.set(REDIS_KEY_STATE, state)
    return jsonify({'spots': res_spots, 'completed': True})


@app.route('/reset/flow', methods=['POST'])
def reset_flow_models():
    if all(ctype not in request.headers['Content-Type'] for ctype
           in ('multipart/form-data', 'application/json')):
        msg = 'Content-Type should be multipart/form-data or application/json'
        logger().error(msg)
        return (jsonify(error=msg), 400)
    if int(redis_client.get(REDIS_KEY_STATE)) != TrainState.IDLE.value:
        return jsonify(error='Process is running'), 500
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
        req_json['device'] = device()
        config = ResetConfig(req_json)
        logger().info(config)
        init_flow_models(config.model_path,
                         config.device,
                         config.is_3d,
                         url=config.url,
                         state_dicts=state_dicts)
    except RuntimeError as e:
        logger().exception('Failed in reset_flow_models')
        return jsonify(error=f'Runtime Error: {e}'), 500
    except Exception as e:
        logger().exception('Failed in reset_flow_models')
        return jsonify(error=f'Exception: {e}'), 500
    return jsonify({'completed': True})


def _dilate_2d_indices(rr, cc, shape):
    if len(rr) != len(cc):
        raise RuntimeError('indices should have the same length')
    n_pixels = len(rr)
    rr_dilate = np.array([0, ] * (n_pixels * 3 ** 2))
    cc_dilate = np.copy(rr_dilate)
    offset = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            rr_dilate[offset:offset +
                      n_pixels] = (rr + dy).clip(0, shape[0] - 1)
            cc_dilate[offset:offset +
                      n_pixels] = (cc + dx).clip(0, shape[1] - 1)
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
                dd_dilate[offset:offset +
                          n_pixels] = (dd + dz).clip(0, shape[0] - 1)
                rr_dilate[offset:offset +
                          n_pixels] = (rr + dy).clip(0, shape[1] - 1)
                cc_dilate[offset:offset +
                          n_pixels] = (cc + dx).clip(0, shape[2] - 1)
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
        for chunk in Path(za_label.store.path).glob(f'{t}.*'):
            chunk.unlink()
        for chunk in Path(za_label_vis.store.path).glob(f'{t}.*'):
            chunk.unlink()
        label_vis = np.zeros(img_shape + (3,), dtype='uint8')
        cnt = collections.Counter({x: 0 for x in keyorder})
        for spot in spots:
            if int(redis_client.get(REDIS_KEY_STATE)) == TrainState.IDLE.value:
                logger().info('update aborted')
                return jsonify({'completed': False})
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
                label_vis[..., 1][indices_outer] = np.where(
                    cond_outer_1,
                    label_vis_value,
                    label_vis[..., 1][indices_outer]
                )
                label[indices_inner_p] = 2 + label_offset
                label_vis[..., 1][indices_inner_p] = label_vis_value
                cond_inner = np.fmod(label[indices_inner] - 1, 3) <= 2
                label[indices_inner] = np.where(
                    cond_inner,
                    3 + label_offset,
                    label[indices_inner]
                )
                label_vis[..., 2][indices_inner] = np.where(
                    cond_inner,
                    label_vis_value,
                    label_vis[..., 2][indices_inner]
                )
            elif spot['tag'] in ('tb', 'fb'):
                label[indices_outer] = np.where(
                    cond_outer_1,
                    2 + label_offset,
                    label[indices_outer]
                )
                label_vis[..., 1][indices_outer] = np.where(
                    cond_outer_1,
                    label_vis_value,
                    label_vis[..., 1][indices_outer]
                )
            elif spot['tag'] in ('tn', 'fp'):
                cond_outer_0 = np.fmod(label[indices_outer] - 1, 3) <= 0
                label[indices_outer] = np.where(
                    cond_outer_0,
                    1 + label_offset,
                    label[indices_outer]
                )
                label_vis[..., 0][indices_outer] = np.where(
                    cond_outer_0,
                    label_vis_value,
                    label_vis[..., 0][indices_outer]
                )
        logger().info('frame:{}, {}'.format(
            t, sorted(cnt.items(), key=lambda i: keyorder.index(i[0]))))
        if memmap_dir:
            fpath = Path(memmap_dir) / f'{keybase}-t{t}-seglabel.dat'
            lock = FileLock(str(fpath) + '.lock')
            with lock:
                if fpath.exists():
                    logger().info(f'remove {fpath}')
                    fpath.unlink()
        target = tuple(np.array(list(label_indices)).T)
        target_t = to_fancy_index(t, *target)
        target_vis = tuple(
            np.column_stack([to_fancy_index(*target, c) for c in range(3)])
        )
        target_vis_t = to_fancy_index(t, *target_vis)
        za_label[target_t] = label[target]
        za_label_vis[target_vis_t] = label_vis[target_vis]
        za_label.attrs[f'label.indices.{t}'] = list(label_indices)
        za_label.attrs['updated'] = True
        if is_livemode:
            if redis_client.get(REDIS_KEY_TIMEPOINT):
                msg = 'Last update/training is ongoing'
                logger().error(msg)
                return jsonify(error=msg), 400
            redis_client.set(REDIS_KEY_TIMEPOINT, t)
    return jsonify({'completed': True})


@app.route('/update/seg', methods=['POST'])
def update_seg_labels():
    if request.headers['Content-Type'] != 'application/json':
        msg = 'Content-Type should be application/json'
        logger().error(msg)
        return jsonify(error=msg), 400
    state = int(redis_client.get(REDIS_KEY_STATE))
    while (state == TrainState.WAIT.value):
        logger().info(f'waiting @{request.path}')
        time.sleep(1)
        state = int(redis_client.get(REDIS_KEY_STATE))
        if (state == TrainState.IDLE.value):
            return jsonify({'completed': False})
    if redis_client.get(REDIS_KEY_UPDATE_ONGOING_SEG):
        msg = 'Last update is ongoing'
        logger().error(msg)
        return jsonify(error=msg), 400
    try:
        redis_client.set(REDIS_KEY_UPDATE_ONGOING_SEG, 1)
        redis_client.set(REDIS_KEY_STATE, TrainState.WAIT.value)
        req_json = request.get_json()
        req_json['device'] = device()
        config = SegmentationTrainConfig(req_json)
        logger().info(config)
        if config.is_livemode and redis_client.get(REDIS_KEY_TIMEPOINT):
            msg = 'Last update/training is ongoing'
            logger().error(msg)
            return jsonify(error=msg), 400
        if req_json.get('reset'):
            try:
                zarr.open_like(zarr.open(config.zpath_seg_label, mode='r'),
                               config.zpath_seg_label,
                               mode='w')
                zarr.open_like(zarr.open(config.zpath_seg_label_vis, mode='r'),
                               config.zpath_seg_label_vis,
                               mode='w')
            except RuntimeError as e:
                logger().exception('Failed in opening zarr')
                return jsonify(error=f'Runtime Error: {e}'), 500
            except Exception as e:
                logger().exception('Failed in opening zarr')
                return jsonify(error=f'Exception: {e}'), 500
            return jsonify({'completed': True})

        spots_dict = collections.defaultdict(list)
        for spot in req_json.get('spots'):
            spots_dict[spot['t']].append(spot)
        if not spots_dict:
            msg = 'nothing to update'
            logger().error(msg)
            return jsonify(error=msg), 400
        if config.is_livemode and len(spots_dict.keys()) != 1:
            msg = 'Livemode should update only a single timepoint'
            logger().error(msg)
            return jsonify(error=msg), 400
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
        except RuntimeError as e:
            logger().exception('Failed in _update_seg_labels')
            return jsonify(error=f'Runtime Error: {e}'), 500
        except Exception as e:
            logger().exception('Failed in _update_seg_labels')
            return jsonify(error=f'Exception: {e}'), 500
    finally:
        if int(redis_client.get(REDIS_KEY_STATE)) != TrainState.IDLE.value:
            redis_client.set(REDIS_KEY_STATE, state)
        redis_client.delete(REDIS_KEY_UPDATE_ONGOING_SEG)
    return response


@app.route('/train/seg', methods=['POST'])
def train_seg():
    if request.headers['Content-Type'] != 'application/json':
        msg = 'Content-Type should be application/json'
        logger().error(msg)
        return jsonify(error=msg), 400
    try:
        req_json = request.get_json()
        req_json['device'] = device()
        config = SegmentationTrainConfig(req_json)
        logger().info(config)
        if config.n_crops < 1:
            msg = 'n_crops should be a positive number'
            logger().error(msg)
            return jsonify(error=msg), 400

        spots_dict = collections.defaultdict(list)
        for spot in req_json.get('spots'):
            spots_dict[spot['t']].append(spot)
        if not (spots_dict or config.is_livemode):
            msg = 'nothing to train'
            logger().error(msg)
            return jsonify(error=msg), 400
        spots_dict = collections.OrderedDict(sorted(spots_dict.items()))

        if int(redis_client.get(REDIS_KEY_STATE)) != TrainState.IDLE.value:
            msg = 'Process is running'
            logger().error(msg)
            return jsonify(error=msg), 500
        redis_client.set(REDIS_KEY_STATE, TrainState.RUN.value)
        if config.is_livemode:
            redis_client.delete(REDIS_KEY_TIMEPOINT)
        else:
            _update_seg_labels(spots_dict,
                               config.scales,
                               config.zpath_input,
                               config.zpath_seg_label,
                               config.zpath_seg_label_vis,
                               config.auto_bg_thresh,
                               config.c_ratio,
                               memmap_dir=config.memmap_dir)
        step_offset = 0
        for path in sorted(Path(config.log_dir).glob('event*')):
            try:
                *_, last_record = TFRecordDataset(str(path))
                last = event_pb2.Event.FromString(last_record.numpy()).step
                step_offset = max(step_offset, last+1)
            except Exception:
                pass
        epoch_start = 0
        train_seg_task.delay(
            list(spots_dict.keys()), config.batch_size,
            config.crop_size, config.class_weights,
            config.false_weight, config.model_path, config.n_epochs,
            config.keep_axials, config.scales, config.lr,
            config.n_crops, config.is_3d, config.is_livemode,
            config.scale_factor_base, config.rotation_angle, config.contrast,
            config.zpath_input, config.zpath_seg_label,
            config.log_interval, config.log_dir, step_offset, epoch_start,
            config.is_cpu(), config.is_mixed_precision, config.cache_maxbytes,
            config.memmap_dir,
        ).wait()
        if (redis_client is not None and
                int(redis_client.get(REDIS_KEY_STATE))
                == TrainState.IDLE.value):
            logger().info('training aborted')
            return jsonify({'completed': False})
    except KeyboardInterrupt:
        logger().info('training aborted')
        return jsonify({'completed': False})
    except RuntimeError as e:
        logger().exception('Failed in train_seg')
        return jsonify(error=f'Runtime Error: {e}'), 500
    except Exception as e:
        logger().exception('Failed in train_seg')
        return jsonify(error=f'Exception: {e}'), 500
    finally:
        torch.cuda.empty_cache()
        redis_client.set(REDIS_KEY_STATE, TrainState.IDLE.value)
    return jsonify({'completed': True})


@app.route('/predict/seg', methods=['POST'])
def predict_seg():
    if request.headers['Content-Type'] != 'application/json':
        msg = 'Content-Type should be application/json'
        return jsonify(error=msg), 400
    state = int(redis_client.get(REDIS_KEY_STATE))
    while (state == TrainState.WAIT.value):
        logger().info(f'waiting @{request.path}')
        time.sleep(1)
        state = int(redis_client.get(REDIS_KEY_STATE))
        if (state == TrainState.IDLE.value):
            return jsonify({'completed': False})
    try:
        req_json = request.get_json()
        req_json['device'] = device()
        config = SegmentationEvalConfig(req_json)
        logger().info(config)
        redis_client.set(REDIS_KEY_STATE, TrainState.WAIT.value)
        spots = detect_spots_task.delay(
            str(config.device), config.model_path, config.keep_axials,
            config.is_3d, config.crop_size, config.scales,
            config.cache_maxbytes, config.use_2d, config.use_median,
            config.patch_size, config.crop_box, config.c_ratio, config.p_thresh,
            config.r_min, config.r_max, config.output_prediction,
            config.zpath_input, config.zpath_seg_output, config.timepoint,
            None, config.memmap_dir, config.batch_size,
        ).wait()
        if (redis_client is not None and
                int(redis_client.get(REDIS_KEY_STATE))
                == TrainState.IDLE.value):
            logger().info('prediction aborted')
            return jsonify({'spots': [], 'completed': False})
        publish_mq('prediction', 'Prediction updated')
    except KeyboardInterrupt:
        logger().info('prediction aborted')
        return jsonify({'spots': [], 'completed': False})
    except RuntimeError as e:
        logger().exception('Failed in detect_spots')
        return jsonify(error=f'Runtime Error: {e}'), 500
    except Exception as e:
        logger().exception('Failed in detect_spots')
        return jsonify(error=f'Exception: {e}'), 500
    finally:
        gc.collect()
        torch.cuda.empty_cache()
        if int(redis_client.get(REDIS_KEY_STATE)) != TrainState.IDLE.value:
            redis_client.set(REDIS_KEY_STATE, state)
    return jsonify({'spots': spots, 'completed': True})


@app.route('/reset/seg', methods=['POST'])
def reset_seg_models():
    if all(ctype not in request.headers['Content-Type'] for ctype
           in ('multipart/form-data', 'application/json')):
        msg = 'Content-Type should be multipart/form-data or application/json'
        logger().error(msg)
        return (jsonify(error=msg), 400)
    if int(redis_client.get(REDIS_KEY_STATE)) != TrainState.IDLE.value:
        msg = 'Process is running. Model cannot be reset.'
        logger().error(msg)
        return jsonify(error=msg), 500
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
        req_json['device'] = device()
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
        return jsonify(error=f'Runtime Error: {e}'), 500
    except Exception as e:
        logger().exception('Failed in init_seg_models')
        return jsonify(error=f'Exception: {e}'), 500
    finally:
        gc.collect()
        torch.cuda.empty_cache()
        redis_client.set(REDIS_KEY_STATE, TrainState.IDLE.value)
    return jsonify({'completed': True})


@app.route('/export/ctc', methods=['POST'])
def export_ctc():
    if request.headers['Content-Type'] != 'application/json':
        msg = 'Content-Type should be application/json'
        return jsonify(error=msg), 400
    state = int(redis_client.get(REDIS_KEY_STATE))
    while (state == TrainState.WAIT.value):
        logger().info(f'waiting @{request.path}')
        time.sleep(1)
        state = int(redis_client.get(REDIS_KEY_STATE))
        if (state == TrainState.IDLE.value):
            return make_response('', 204)
    try:
        req_json = request.get_json()
        req_json['device'] = device()
        config = ExportConfig(req_json)
        za_input = zarr.open(config.zpath_input, mode='r')
        config.shape = za_input.shape[1:]
        logger().info(config)
        spots_dict = collections.defaultdict(list)
        for spot in req_json.get('spots'):
            spots_dict[spot['t']].append(spot)
        spots_dict = collections.OrderedDict(sorted(spots_dict.items()))

        redis_client.set(REDIS_KEY_STATE, TrainState.WAIT.value)
        result = export_ctc_labels(config, spots_dict, redis_client)
        if isinstance(result, str):
            resp = send_file(result)
            # file_remover.cleanup_once_done(resp, result)
        elif not result:
            resp = make_response('', 204)
        else:
            resp = make_response('', 200)
    except RuntimeError as e:
        logger().exception('Failed in export_ctc_labels')
        return jsonify(error=f'Runtime Error: {e}'), 500
    except Exception as e:
        logger().exception('Failed in export_ctc_labels')
        return jsonify(error=f'Exception: {e}'), 500
    finally:
        if int(redis_client.get(REDIS_KEY_STATE)) != TrainState.IDLE.value:
            redis_client.set(REDIS_KEY_STATE, state)
    return resp


@app.route('/gpus', methods=['GET'])
def get_gpus():
    gpus = []
    try:
        for gpu in nvsmi.get_gpus():
            gpus.append({
                'id': gpu.id,
                'name': gpu.name,
                'mem_total': gpu.mem_total,
                'mem_used': gpu.mem_used
            })
        resp = jsonify(gpus)
    except Exception:
        pass
    if len(gpus) == 0:
        resp = jsonify([{
            'id': 'GPU is not available',
            'name': 'CPU is used',
            'mem_total': psutil.virtual_memory().total / 1024 / 1024,
            'mem_used': psutil.virtual_memory().used / 1024 / 1024
        }])
    return resp


@app.route('/dataset/check', methods=['POST'])
def check_datset():
    if request.headers['Content-Type'] != 'application/json':
        msg = 'Content-Type should be application/json'
        logger().error(msg)
        return jsonify(error=msg), 400
    req_json = request.get_json()
    if 'dataset_name' not in req_json:
        return jsonify(error='dataset_name key is missing'), 400
    if 'shape' not in req_json:
        return jsonify(error='shape key is missing'), 400
    message = dstool.check_dataset(
        Path(DATASETS_DIR) / req_json['dataset_name'],
        tuple(req_json['shape'])
    )
    return jsonify(message=message), 200


@app.route('/dataset/generate', methods=['POST'])
def gen_datset():
    if request.headers['Content-Type'] != 'application/json':
        msg = 'Content-Type should be application/json'
        logger().error(msg)
        return jsonify(error=msg), 400
    req_json = request.get_json()
    if 'dataset_name' not in req_json:
        return jsonify(error='dataset_name key is missing'), 400
    p_dataset = Path(DATASETS_DIR) / req_json['dataset_name']
    h5_files = list(sorted(p_dataset.glob('*.h5')))
    if len(h5_files) == 0:
        logger().info(f'.h5 file not found @{request.path}')
        return jsonify(
            message=f'.h5 file not found in {req_json["dataset_name"]}'), 204
    elif 1 < len(h5_files):
        logger().info(
            f'multiple .h5 files found, use the first one {h5_files[0]}')
    try:
        dstool.generate_dataset(
            h5_files[0],
            p_dataset,
            req_json.get('is_uint16', None),
            req_json.get('divisor', 1.),
            req_json.get('is_2d', False),
            True,
        )
    except Exception as e:
        logger().exception('Failed in gen_datset')
        return jsonify(error=f'Exception: {e}'), 500
    return make_response('', 200)


@app.route('/upload', methods=['POST'])
def upload():
    """ Upload an image data to the dataset directory. # noqa: E501

    See details:
        https://github.com/pallets/flask/issues/2086
    Code from:
        https://gitlab.nsd.no/ire/python-webserver-file-submission-poc/blob/master/flask_app.py

    Expected HTTP request:
        POST /upload HTTP/1.1
        Host: localhost:8080
        Content-Type: multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW
        Content-Length: <Content-Length here>

        ----WebKitFormBoundary7MA4YWxkTrZu0gW
        Content-Disposition: form-data; name="file"; filename="YOUR_IMAGE_FILE.h5"
        Content-Type: <Content-Type header here>

        (data)
        ----WebKitFormBoundary7MA4YWxkTrZu0gW
        Content-Disposition: form-data; name="dataset"

        YOUR_DATASET_NAME
        ----WebKitFormBoundary7MA4YWxkTrZu0gW

    """
    tmpfile, tmpfile_path = tempfile.mkstemp(prefix='elephant')
    with os.fdopen(tmpfile, 'w+b') as f:
        def custom_stream_factory(total_content_length, filename,
                                  content_type, content_length=None):
            return f
        _, form, _ = werkzeug.formparser.parse_form_data(
            request.environ, stream_factory=custom_stream_factory)

        if 'dataset_name' not in form:
            return jsonify(res='dataset_name is not specified'), 400
        if 'filename' not in form:
            return jsonify(res='filename is not specified'), 400
        if 'action' not in form:
            return jsonify(res='action is not specified'), 400
        if form['action'] not in ('init', 'append', 'complete', 'cancel'):
            return jsonify(res=f'unknown action: {form["action"]}'), 400
        p_file = Path(DATASETS_DIR) / form['dataset_name'] / form['filename']
        p_file.parent.mkdir(parents=True, exist_ok=True)
        p_file_tmp = p_file.with_suffix('.tmp')
        if form['action'] == 'complete':
            p_file_tmp.rename(p_file)
        elif form['action'] in ('init', 'cancel') and p_file_tmp.exists():
            p_file_tmp.unlink()
        if form['action'] in ('init', 'append'):
            f.seek(0)
            with p_file_tmp.open('ab') as dest_file:
                dest_file.write(f.read())
            Path(tmpfile_path).unlink()
    return make_response('', 200)


@app.route('/model/download', methods=['POST'])
def download_model():
    """ Download a model paramter file.
    """
    if request.headers['Content-Type'] != 'application/json':
        msg = 'Content-Type should be application/json'
        logger().error(msg)
        return jsonify(error=msg), 400
    req_json = request.get_json()
    config = BaseConfig(req_json)
    logger().info(config)
    if not Path(config.model_path).exists():
        logger().info(
            f'model file {config.model_path} not found @{request.path}')
        return jsonify(
            message=f'model file {config.model_path} not found'), 204
    try:
        resp = send_file(config.model_path)
    except Exception as e:
        logger().exception(
            'Failed to prepare a model parameter file for download')
        return jsonify(error=f'Exception: {e}'), 500
    return resp


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=80)
