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

from elephant.common import init_flow_models
from elephant.common import run_train_flow
from elephant.common import spots_with_flow
from elephant.config import FlowEvalConfig
from elephant.config import FlowTrainConfig
from elephant.config import ResetConfig
from elephant.logging import logger
from elephant.redis_util import get_state
from elephant.redis_util import redis_client
from elephant.redis_util import TrainState
from elephant.redis_util import REDIS_KEY_STATE
from elephant.redis_util import REDIS_KEY_UPDATE_ONGOING_FLOW
from elephant.util import get_device
from elephant.util.ellipse import ellipse
from elephant.util.ellipsoid import ellipsoid

api = Namespace('flow', description='Flow APIs')


@shared_task()
def train_flow_task(spot_indices, batch_size, crop_size, model_path, n_epochs,
                    keep_axials, scales, lr, n_crops, is_3d, scale_factor_base,
                    rotation_angle, zpath_input, zpath_flow_label,
                    log_interval, log_dir, step_offset=0, epoch_start=0,
                    is_cpu=False, is_mixed_precision=False, cache_maxbytes=None,
                    memmap_dir=None, input_size=None):
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
                       cache_maxbytes, memmap_dir, input_size),
                 nprocs=world_size,
                 join=True)
    else:
        run_train_flow(torch.device('cpu') if is_cpu else torch.device('cuda'),
                       world_size, spot_indices, batch_size, crop_size,
                       model_path, n_epochs, keep_axials, scales, lr,
                       n_crops, is_3d, scale_factor_base, rotation_angle,
                       zpath_input, zpath_flow_label, log_interval, log_dir,
                       step_offset, epoch_start, is_cpu, is_mixed_precision,
                       cache_maxbytes, memmap_dir, input_size)


@shared_task()
def spots_with_flow_task(device, spots, model_path, keep_axials=(True,) * 4,
                         is_pad=False, is_3d=True, scales=None,
                         use_median=False, patch_size=None, crop_box=None,
                         output_prediction=False, zpath_input=None,
                         zpath_flow=None, timepoint=None, tiff_input=None,
                         memmap_dir=None, batch_size=1, input_size=None,
                         flow_norm_factor=None):
    return spots_with_flow(device, spots, model_path, keep_axials, is_pad,
                           is_3d, scales, use_median, patch_size,
                           crop_box, output_prediction, zpath_input,
                           zpath_flow, timepoint, tiff_input, memmap_dir,
                           batch_size, tuple(input_size), flow_norm_factor)


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
        centroids = []
        for spot in spots:
            if get_state() == TrainState.IDLE.value:
                logger().info('update aborted')
                return make_response(jsonify({'completed': False}))
            centroid = np.array(spot['pos'][::-1])
            centroid = centroid[-n_dims:]
            centroids.append((centroid / scales).astype(int).tolist())
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
                ind = (np.full(len(indices[0]), i),) + indices
                label[ind] = (
                    displacement[i] / scales[-1 - i] / flow_norm_factor[i])
            # last channels is for weight
            ind = (np.full(len(indices[0]), -1),) + indices
            label[ind] = weight
            label_indices.update(
                tuple(map(tuple, np.stack(indices, axis=1).tolist()))
            )
        logger().info(f'frame:{t+1}, {len(spots)} linkings')
        target = tuple(np.array(list(label_indices)).T)
        target = (
            np.array(
                sum(
                    tuple([(i,) * len(target[0]) for i in range(n_dims + 1)]),
                    ()
                )
            ),
        ) + tuple(np.tile(target[i], n_dims + 1) for i in range(n_dims))
        target_t = (np.full(len(target[0]), t),) + target
        za_label.attrs[f'label.indices.{t}'] = centroids
        za_label.attrs['updated'] = True
        za_label[target_t] = label[target]
    return make_response(jsonify({'completed': True}))


@ api.route('/train')
class Train(Resource):
    @ api.doc()
    def post(self):
        '''
        Train flow model.

        '''
        if request.headers['Content-Type'] != 'application/json':
            msg = 'Content-Type should be application/json'
            logger().error(msg)
            return make_response(jsonify(error=msg), 400)
        try:
            req_json = request.get_json()
            req_json['device'] = get_device()
            config = FlowTrainConfig(req_json)
            logger().info(config)
            if config.n_crops < 1:
                msg = 'n_crops should be a positive number'
                logger().error(msg)
                return make_response(jsonify(error=msg), 400)

            spots_dict = collections.defaultdict(list)
            for spot in req_json.get('spots'):
                spots_dict[spot['t']].append(spot)
            if not spots_dict:
                msg = 'nothing to train'
                logger().error(msg)
                return make_response(jsonify(error=msg), 400)
            spots_dict = collections.OrderedDict(sorted(spots_dict.items()))

            if get_state() != TrainState.IDLE.value:
                msg = 'Process is running'
                logger().error(msg)
                return make_response(jsonify(error='Process is running'), 500)
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
            async_result = train_flow_task.delay(
                list(spots_dict.keys()), config.batch_size, config.crop_size,
                config.model_path, config.n_epochs, config.keep_axials,
                config.scales, config.lr, config.n_crops, config.is_3d,
                config.scale_factor_base, config.rotation_angle,
                config.zpath_input, config.zpath_flow_label,
                config.log_interval, config.log_dir, step_offset, epoch_start,
                config.is_cpu(), False,
                config.cache_maxbytes, config.memmap_dir, config.input_size
            )
            while not async_result.ready():
                if (redis_client is not None and
                    get_state()
                        == TrainState.IDLE.value):
                    logger().info('training aborted')
                    return make_response(jsonify({'completed': False}))
        except Exception as e:
            logger().exception('Failed in train_flow')
            return make_response(jsonify(error=f'Exception: {e}'), 500)
        finally:
            gc.collect()
            torch.cuda.empty_cache()
            redis_client.set(REDIS_KEY_STATE, TrainState.IDLE.value)
        return make_response(jsonify({'completed': True}))


@ api.route('/update')
class Update(Resource):
    @ api.doc()
    def post(self):
        '''
        Update flow label.

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
        if redis_client.get(REDIS_KEY_UPDATE_ONGOING_FLOW):
            msg = 'Last update is ongoing'
            logger().error(msg)
            return make_response(jsonify(error=msg), 400)
        try:
            redis_client.set(REDIS_KEY_STATE, TrainState.WAIT.value)
            req_json = request.get_json()
            req_json['device'] = get_device()
            config = FlowTrainConfig(req_json)
            logger().info(config)
            if req_json.get('reset'):
                zarr.open_like(zarr.open(config.zpath_flow_label, mode='r'),
                               config.zpath_flow_label,
                               mode='w')
                return make_response(jsonify({'completed': True}))

            spots_dict = collections.defaultdict(list)
            for spot in req_json.get('spots'):
                spots_dict[spot['t']].append(spot)
            if not spots_dict:
                msg = 'nothing to update'
                logger().error(msg)
                return make_response(jsonify(error=msg), 400)
            spots_dict = collections.OrderedDict(sorted(spots_dict.items()))

            response = _update_flow_labels(spots_dict,
                                           config.scales,
                                           config.zpath_flow_label,
                                           config.flow_norm_factor)
        except RuntimeError as e:
            logger().exception('Failed in update_flow_labels')
            return make_response(jsonify(error=f'Runtime Error: {e}'), 500)
        except Exception as e:
            logger().exception('Failed in update_flow_labels')
            return make_response(jsonify(error=f'Exception: {e}'), 500)
        finally:
            if get_state() != TrainState.IDLE.value:
                redis_client.set(REDIS_KEY_STATE, state)
            redis_client.delete(REDIS_KEY_UPDATE_ONGOING_FLOW)
        return response


@ api.route('/predict')
class Predict(Resource):
    @ api.doc()
    def post(self):
        '''
        Predict flow.

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
                logger().info(f'{request.path} cancelled')
                return make_response(jsonify({'completed': False}))
        try:
            req_json = request.get_json()
            req_json['device'] = get_device()
            config = FlowEvalConfig(req_json)
            logger().info(config)
            redis_client.set(REDIS_KEY_STATE, TrainState.WAIT.value)
            spots = req_json.get('spots')
            async_result = spots_with_flow_task.delay(
                str(config.device), spots, config.model_path,
                config.keep_axials, config.is_pad, config.is_3d, config.scales,
                config.use_median, config.patch_size, config.crop_box,
                config.output_prediction, config.zpath_input,
                config.zpath_flow, config.timepoint, None, config.memmap_dir,
                config.batch_size, config.input_size, config.flow_norm_factor,
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
            res_spots = async_result.result
            if res_spots is None:
                logger().info('prediction aborted')
                return make_response(jsonify({'spots': [], 'completed': False}))
        except Exception as e:
            logger().exception('Failed in predict_flow')
            return make_response(jsonify(error=f'Exception: {e}'), 500)
        finally:
            gc.collect()
            torch.cuda.empty_cache()
            if get_state() != TrainState.IDLE.value:
                redis_client.set(REDIS_KEY_STATE, state)
        return make_response(jsonify({'spots': res_spots, 'completed': True}))


@ api.route('/reset')
class Reset(Resource):
    @ api.doc()
    def post(self):
        '''
        Reset flow model.

        '''
        if all(ctype not in request.headers['Content-Type'] for ctype
               in ('multipart/form-data', 'application/json')):
            msg = ('Content-Type should be multipart/form-data or '
                   'application/json')
            logger().error(msg)
            return (jsonify(error=msg), 400)
        if get_state() != TrainState.IDLE.value:
            return make_response(jsonify(error='Process is running'), 500)
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
            init_flow_models(config.model_path,
                             config.device,
                             config.is_3d,
                             url=config.url,
                             state_dicts=state_dicts)
        except RuntimeError as e:
            logger().exception('Failed in reset_flow_models')
            return make_response(jsonify(error=f'Runtime Error: {e}'), 500)
        except Exception as e:
            logger().exception('Failed in reset_flow_models')
            return make_response(jsonify(error=f'Exception: {e}'), 500)
        return make_response(jsonify({'completed': True}))
