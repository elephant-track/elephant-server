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
import os
from pathlib import Path
import tempfile
import weakref

from flask import Flask
from flask import g
from flask import jsonify
from flask import make_response
from flask import request
from flask import send_file
from flask_redis import FlaskRedis
import numpy as np
import nvsmi
import pika
from tensorflow.data import TFRecordDataset
from tensorflow.core.util import event_pb2
import time
import torch
import torch.utils.data as du
import werkzeug
import zarr

from elephant.common import TensorBoard
from elephant.common import detect_spots
from elephant.common import export_ctc_labels
from elephant.common import init_flow_models
from elephant.common import init_seg_models
from elephant.common import load_flow_models
from elephant.common import load_seg_models
from elephant.common import spots_with_flow
from elephant.common import train
from elephant.config import DATASETS_DIR
from elephant.config import ExportConfig
from elephant.config import FlowEvalConfig
from elephant.config import FlowTrainConfig
from elephant.config import ResetConfig
from elephant.config import SegmentationEvalConfig
from elephant.config import SegmentationTrainConfig
from elephant.datasets import FlowDatasetZarr
from elephant.datasets import SegmentationDatasetZarr
from elephant.logging import logger
from elephant.losses import FlowLoss
from elephant.losses import SegmentationLoss
from elephant.redis_util import REDIS_KEY_COUNT
from elephant.redis_util import REDIS_KEY_LR
from elephant.redis_util import REDIS_KEY_NCROPS
from elephant.redis_util import REDIS_KEY_STATE
from elephant.redis_util import REDIS_KEY_TIMEPOINT
from elephant.redis_util import TrainState
from elephant.tool import dataset as dstool
from elephant.util import normalize_zero_one
from elephant.util import get_device
from elephant.util.ellipse import ellipse
from elephant.util.ellipsoid import ellipsoid

REDIS_URL = "redis://:password@localhost:6379/0"

app = Flask(__name__)
redis_client = FlaskRedis(app)
redis_client.set(REDIS_KEY_STATE, TrainState.IDLE.value)
redis_client.set(REDIS_KEY_COUNT, 0)


# https: // stackoverflow.com/a/32132035
class FileRemover(object):
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


def get_mq_connection():
    if 'connection' not in g:
        g.connection = pika.BlockingConnection(
            pika.ConnectionParameters(host='localhost', heartbeat=0))
        g.connection.channel().queue_declare(queue='update')
        g.connection.channel().queue_declare(queue='dataset')
        g.connection.channel().queue_declare(queue='log')
        g.connection.channel().queue_declare(queue='prediction')
    if g.connection.is_closed:
        g.connection.open()
    return g.connection


for _ in range(10):
    try:
        with app.app_context():
            logger().handlers[0].connection_provider = get_mq_connection
            print('set up MQ connection')
        break
    except Exception:
        print('waiting for RabbitMQ...')
    time.sleep(2)


@app.teardown_appcontext
def teardown_mq_connection(error):
    connection = g.pop('connection', None)
    if connection is not None and connection.is_open:
        connection.close()


def device():
    if 'device' not in g:
        g.device = get_device()
    return g.device


@app.route('/state', methods=['GET', 'POST'])
def state():
    if request.method == 'POST':
        if request.headers['Content-Type'] != 'application/json':
            return jsonify(error='Content-Type should be application/json'), 400
        req_json = request.get_json()
        state = req_json.get(REDIS_KEY_STATE)
        if (not isinstance(state, int) or
                state not in TrainState._value2member_map_):
            return jsonify(res=f'Invalid state: {state}'), 400
        redis_client.set(REDIS_KEY_STATE, state)
    return jsonify(success=True, state=int(redis_client.get(REDIS_KEY_STATE)))


@app.route('/params', methods=['GET', 'POST'])
def params():
    if request.method == 'POST':
        if request.headers['Content-Type'] != 'application/json':
            return jsonify(error='Content-Type should be application/json'), 400
        req_json = request.get_json()
        lr = req_json.get('lr')
        if not isinstance(lr, float) or lr < 0:
            return jsonify(res=f'Invalid learning rate: {lr}'), 400
        n_crops = req_json.get('n_crops')
        if not isinstance(n_crops, int) or n_crops < 0:
            return jsonify(res=f'Invalid number of crops: {n_crops}'), 400
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
        for spot in spots:
            if int(redis_client.get(REDIS_KEY_STATE)) == TrainState.IDLE.value:
                logger().info('aborted')
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
        logger().info(f'frame:{t+1}, {len(spots)} linkings')
        za_label[t] = label
    return jsonify({'completed': True})


@app.route('/update/flow', methods=['POST'])
def update_flow_labels():
    if request.headers['Content-Type'] != 'application/json':
        return jsonify(error='Content-Type should be application/json'), 400
    state = int(redis_client.get(REDIS_KEY_STATE))
    while (state == TrainState.WAIT.value):
        logger().info("waiting", "@/update/flow")
        time.sleep(1)
        state = int(redis_client.get(REDIS_KEY_STATE))
        if (state == TrainState.IDLE.value):
            return jsonify({'completed': False})
    try:
        redis_client.set(REDIS_KEY_STATE, TrainState.WAIT.value)
        req_json = request.get_json()
        req_json['device'] = device()
        config = FlowTrainConfig(req_json)
        logger().info(config)
        if req_json.get('reset'):
            try:
                zarr.open_like(zarr.open(config.zpath_flow_label, mode='r'),
                               config.zpath_flow_label,
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
        spots_dict = collections.OrderedDict(sorted(spots_dict.items()))

        try:
            response = _update_flow_labels(spots_dict,
                                           config.scales,
                                           config.zpath_flow_label,
                                           config.flow_norm_factor)
        except RuntimeError as e:
            logger().exception('Failed in _update_flow_labels')
            return jsonify(error=f'Runtime Error: {e}'), 500
        except Exception as e:
            logger().exception('Failed in _update_flow_labels')
            return jsonify(error=f'Exception: {e}'), 500
    finally:
        if int(redis_client.get(REDIS_KEY_STATE)) != TrainState.IDLE.value:
            redis_client.set(REDIS_KEY_STATE, state)
    return response


@app.route('/train/flow', methods=['POST'])
def train_flow():
    if request.headers['Content-Type'] != 'application/json':
        return jsonify(error='Content-Type should be application/json'), 400
    req_json = request.get_json()
    req_json['device'] = device()
    config = FlowTrainConfig(req_json)
    logger().info(config)
    spots_dict = collections.defaultdict(list)
    for spot in req_json.get('spots'):
        spots_dict[spot['t']].append(spot)
    spots_dict = collections.OrderedDict(sorted(spots_dict.items()))
    try:
        models = load_flow_models(config.model_path,
                                  config.device,
                                  is_3d=config.is_3d)
    except Exception as e:
        logger().exception('Failed in init_flow_models')
        return jsonify(error=f'Exception: {e}'), 500
    if int(redis_client.get(REDIS_KEY_STATE)) != TrainState.IDLE.value:
        return jsonify(error='Process is running'), 500
    try:
        redis_client.set(REDIS_KEY_STATE, TrainState.RUN.value)
        _update_flow_labels(spots_dict,
                            config.scales,
                            config.zpath_flow_label,
                            config.flow_norm_factor)
        n_dims = 2 + config.is_3d  # 3 or 2
        input_shape = zarr.open(config.zpath_input, mode='r').shape[-n_dims:]
        train_dataset = FlowDatasetZarr(
            config.zpath_input,
            config.zpath_flow_label,
            list(spots_dict.keys()),
            input_shape,
            config.crop_size,
            config.n_crops,
            keep_axials=config.keep_axials,
            scales=config.scales,
            scale_factor_base=config.scale_factor_base,
            rotation_angle=config.rotation_angle,
        )
        if 0 < len(train_dataset):
            loss = FlowLoss(is_3d=config.is_3d).to(config.device)
            optimizers = [torch.optim.Adam(
                model.parameters(), lr=config.lr) for model in models]
            step_offset = 0
            for path in sorted(Path(config.log_dir).glob('event*')):
                try:
                    *_, last_record = TFRecordDataset(str(path))
                    last = event_pb2.Event.FromString(last_record.numpy()).step
                    step_offset = max(step_offset, last+1)
                except Exception:
                    pass
            tb_logger = TensorBoard(config.log_dir)
            train_loader = du.DataLoader(
                train_dataset, shuffle=True, batch_size=1)

            for epoch in range(config.n_epochs):
                if epoch == 50:
                    optimizers = [
                        torch.optim.Adam(model.parameters(), lr=config.lr*0.1)
                        for model in models
                    ]
                for model, optimizer in zip(models, optimizers):
                    train(model, config.device,
                          train_loader,
                          optimizer=optimizer,
                          loss_fn=loss,
                          epoch=epoch,
                          log_interval=config.log_interval,
                          tb_logger=tb_logger,
                          redis_client=redis_client,
                          step_offset=step_offset)
                    if (int(redis_client.get(REDIS_KEY_STATE)) ==
                            TrainState.IDLE.value):
                        logger().info('training aborted')
                        return jsonify({'completed': False})
                torch.save(models[0].state_dict() if len(models) == 1 else
                           [model.state_dict() for model in models],
                           config.model_path)
    except RuntimeError as e:
        logger().exception('Failed in train_flow')
        return jsonify(error=f'Runtime Error: {e}'), 500
    except Exception as e:
        logger().exception('Failed in train_flow')
        return jsonify(error=f'Exception: {e}'), 500
    finally:
        del models
        gc.collect()
        torch.cuda.empty_cache()
        redis_client.set(REDIS_KEY_STATE, TrainState.IDLE.value)
    return jsonify({'completed': True})


@app.route('/predict/flow', methods=['POST'])
def predict_flow():
    if request.headers['Content-Type'] != 'application/json':
        return jsonify(error='Content-Type should be application/json'), 400
    req_json = request.get_json()
    req_json['device'] = device()
    config = FlowEvalConfig(req_json)
    logger().info(config)
    res_spots = []
    if 0 < config.timepoint:
        state = int(redis_client.get(REDIS_KEY_STATE))
        while (state == TrainState.WAIT.value):
            logger().info("waiting", "@/predict/flow")
            time.sleep(1)
            state = int(redis_client.get(REDIS_KEY_STATE))
            if (state == TrainState.IDLE.value):
                return jsonify({'completed': False})
        try:
            redis_client.set(REDIS_KEY_STATE, TrainState.WAIT.value)
            spots = req_json.get('spots')
            res_spots = spots_with_flow(config, spots)
        except RuntimeError as e:
            logger().exception('Failed in spots_with_flow')
            return jsonify(error=f'Runtime Error: {e}'), 500
        except Exception as e:
            logger().exception('Failed in spots_with_flow')
            return jsonify(error=f'Exception: {e}'), 500
        finally:
            gc.collect()
            torch.cuda.empty_cache()
            if int(redis_client.get(REDIS_KEY_STATE)) != TrainState.IDLE.value:
                redis_client.set(REDIS_KEY_STATE, state)
    return jsonify({'spots': res_spots, 'completed': True})


@app.route('/reset/flow', methods=['POST'])
def reset_flow_models():
    if request.headers['Content-Type'] != 'application/json':
        return jsonify(error='Content-Type should be application/json'), 400
    req_json = request.get_json()
    req_json['device'] = device()
    config = ResetConfig(req_json)
    logger().info(config)
    try:
        init_flow_models(config.model_path,
                         config.device,
                         config.is_3d,
                         url=config.url)
    except RuntimeError as e:
        logger().exception('Failed in init_flow_models')
        return jsonify(error=f'Runtime Error: {e}'), 500
    except Exception as e:
        logger().exception('Failed in init_flow_models')
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
                       zpath_seg_label_vis, auto_bg_thresh=0, c_ratio=0.5):
    za_input = zarr.open(zpath_input, mode='r')
    za_label = zarr.open(zpath_seg_label, mode='a')
    za_label_vis = zarr.open(zpath_seg_label_vis, mode='a')
    keyorder = ['tp', 'fp', 'tn', 'fn', 'tb', 'fb']
    MIN_AREA_ELLIPSOID = 9
    n_dims = len(za_input.shape) - 1
    for t, spots in spots_dict.items():
        # label = np.zeros(label_shape, dtype='int64') - 1
        label = np.where(
            normalize_zero_one(za_input[t].astype('float32')) < auto_bg_thresh,
            1,
            0
        ).astype('uint8')
        cnt = collections.Counter({x: 0 for x in keyorder})
        for spot in spots:
            if int(redis_client.get(REDIS_KEY_STATE)) == TrainState.IDLE.value:
                logger().info('aborted')
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
                label.shape,
                MIN_AREA_ELLIPSOID
            )
            label_offset = 0 if spot['tag'] in ['tp', 'tb', 'tn'] else 3
            if spot['tag'] in ('tp', 'fn'):
                indices_inner = draw_func(
                    centroid,
                    radii * c_ratio,
                    rotation,
                    scales,
                    label.shape,
                    MIN_AREA_ELLIPSOID
                )
                indices_inner_p = dilate_func(*indices_inner, label.shape)
                label[indices_outer] = np.where(
                    np.fmod(label[indices_outer] - 1, 3) <= 1,
                    2 + label_offset,
                    label[indices_outer]
                )
                label[indices_inner_p] = 2 + label_offset
                label[indices_inner] = np.where(
                    np.fmod(label[indices_inner] - 1, 3) <= 2,
                    3 + label_offset,
                    label[indices_inner]
                )
            elif spot['tag'] in ('tb', 'fb'):
                label[indices_outer] = np.where(
                    np.fmod(label[indices_outer] - 1, 3) <= 1,
                    2 + label_offset,
                    label[indices_outer]
                )
            elif spot['tag'] in ('tn', 'fp'):
                label[indices_outer] = np.where(
                    np.fmod(label[indices_outer] - 1, 3) <= 0,
                    1 + label_offset,
                    label[indices_outer]
                )
        logger().info('frame:{}, {}'.format(
            t, sorted(cnt.items(), key=lambda i: keyorder.index(i[0]))))
        za_label[t] = label
        za_label_vis[t, ..., 0] = np.where(
            label == 1, 255, 0) + np.where(label == 4, 127, 0)
        za_label_vis[t, ..., 1] = np.where(
            label == 2, 255, 0) + np.where(label == 5, 127, 0)
        za_label_vis[t, ..., 2] = np.where(
            label == 3, 255, 0) + np.where(label == 6, 127, 0)
        if redis_client.get(REDIS_KEY_NCROPS) is not None:
            for i in range(int(redis_client.get(REDIS_KEY_NCROPS))):
                redis_client.rpush(REDIS_KEY_TIMEPOINT, str(t))
    return jsonify({'completed': True})


@app.route('/update/seg', methods=['POST'])
def update_seg_labels():
    if request.headers['Content-Type'] != 'application/json':
        return jsonify(error='Content-Type should be application/json'), 400
    state = int(redis_client.get(REDIS_KEY_STATE))
    while (state == TrainState.WAIT.value):
        logger().info("waiting", "@/update/seg")
        time.sleep(1)
        state = int(redis_client.get(REDIS_KEY_STATE))
        if (state == TrainState.IDLE.value):
            return jsonify({'completed': False})
    try:
        redis_client.set(REDIS_KEY_STATE, TrainState.WAIT.value)
        req_json = request.get_json()
        req_json['device'] = device()
        config = SegmentationTrainConfig(req_json)
        logger().info(config)
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
        spots_dict = collections.OrderedDict(sorted(spots_dict.items()))

        try:
            response = _update_seg_labels(spots_dict,
                                          config.scales,
                                          config.zpath_input,
                                          config.zpath_seg_label,
                                          config.zpath_seg_label_vis,
                                          config.auto_bg_thresh,
                                          config.c_ratio)
        except RuntimeError as e:
            logger().exception('Failed in _update_seg_labels')
            return jsonify(error=f'Runtime Error: {e}'), 500
        except Exception as e:
            logger().exception('Failed in _update_seg_labels')
            return jsonify(error=f'Exception: {e}'), 500
    finally:
        if int(redis_client.get(REDIS_KEY_STATE)) != TrainState.IDLE.value:
            redis_client.set(REDIS_KEY_STATE, state)
    return response


@app.route('/train/seg', methods=['POST'])
def train_seg():
    if request.headers['Content-Type'] != 'application/json':
        return jsonify(error='Content-Type should be application/json'), 400
    req_json = request.get_json()
    req_json['device'] = device()
    config = SegmentationTrainConfig(req_json)
    logger().info(config)
    spots_dict = collections.defaultdict(list)
    for spot in req_json.get('spots'):
        spots_dict[spot['t']].append(spot)
    spots_dict = collections.OrderedDict(sorted(spots_dict.items()))

    try:
        models = load_seg_models(config.model_path,
                                 config.keep_axials,
                                 config.device,
                                 is_3d=config.is_3d,
                                 zpath_input=config.zpath_input,
                                 crop_size=config.crop_size,
                                 scales=config.scales)
    except Exception as e:
        logger().exception('Failed in load_seg_models')
        return jsonify(error=f'Exception: {e}'), 500
    if int(redis_client.get(REDIS_KEY_STATE)) != TrainState.IDLE.value:
        return jsonify(error='Process is running'), 500
    try:
        redis_client.set(REDIS_KEY_STATE, TrainState.RUN.value)
        _update_seg_labels(spots_dict,
                           config.scales,
                           config.zpath_input,
                           config.zpath_seg_label,
                           config.zpath_seg_label_vis,
                           config.auto_bg_thresh,
                           config.c_ratio)
        n_dims = 2 + config.is_3d  # 3 or 2
        input_shape = zarr.open(config.zpath_input, mode='r').shape[-n_dims:]
        train_dataset = SegmentationDatasetZarr(
            config.zpath_input,
            config.zpath_seg_label,
            list(spots_dict.keys()),
            input_shape,
            config.crop_size,
            config.n_crops,
            config.keep_axials,
            scales=config.scales,
            is_livemode=config.is_livemode,
            redis_client=redis_client,
            scale_factor_base=config.scale_factor_base,
            rotation_angle=config.rotation_angle,
        )
        if config.is_livemode:
            redis_client.delete(REDIS_KEY_TIMEPOINT)
        if 0 < len(train_dataset):
            weight_tensor = torch.tensor(config.class_weights).float()
            step_offset = 0
            for path in sorted(Path(config.log_dir).glob('event*')):
                try:
                    *_, last_record = TFRecordDataset(str(path))
                    last = event_pb2.Event.FromString(last_record.numpy()).step
                    step_offset = max(step_offset, last+1)
                except Exception:
                    pass
            tb_logger = TensorBoard(config.log_dir)
            loss = SegmentationLoss(class_weights=weight_tensor,
                                    false_weight=config.false_weight,
                                    is_3d=config.is_3d)
            loss = loss.to(config.device)
            optimizers = [torch.optim.Adam(
                model.parameters(), lr=config.lr) for model in models]
            redis_client.set(REDIS_KEY_LR, str(config.lr))
            train_loader = du.DataLoader(
                train_dataset, shuffle=True, batch_size=1)
            epoch = 0
            while config.is_livemode or epoch < config.n_epochs:
                for model, optimizer in zip(models, optimizers):
                    train(model,
                          config.device,
                          train_loader,
                          optimizer=optimizer,
                          loss_fn=loss,
                          epoch=epoch,
                          log_interval=config.log_interval,
                          tb_logger=tb_logger,
                          redis_client=redis_client,
                          step_offset=step_offset)
                    if (int(redis_client.get(REDIS_KEY_STATE)) ==
                            TrainState.IDLE.value):
                        logger().info('training aborted')
                        return jsonify({'completed': False})
                torch.save(models[0].state_dict() if len(models) == 1 else
                           [model.state_dict() for model in models],
                           config.model_path)
                epoch += 1
                if config.is_livemode:
                    get_mq_connection().channel().basic_publish(
                        exchange='',
                        routing_key='update',
                        body='Model updated'
                    )
    except RuntimeError as e:
        logger().exception('Failed in train_seg')
        return jsonify(error=f'Runtime Error: {e}'), 500
    except Exception as e:
        logger().exception('Failed in train_seg')
        return jsonify(error=f'Exception: {e}'), 500
    finally:
        del models
        gc.collect()
        torch.cuda.empty_cache()
        redis_client.set(REDIS_KEY_STATE, TrainState.IDLE.value)
    return jsonify({'completed': True})


@app.route('/predict/seg', methods=['POST'])
def predict_seg():
    if request.headers['Content-Type'] != 'application/json':
        return jsonify(error='Content-Type should be application/json'), 400
    req_json = request.get_json()
    req_json['device'] = device()
    config = SegmentationEvalConfig(req_json)
    logger().info(config)
    state = int(redis_client.get(REDIS_KEY_STATE))
    while (state == TrainState.WAIT.value):
        logger().info("waiting", "@/predict/seg")
        time.sleep(1)
        state = int(redis_client.get(REDIS_KEY_STATE))
        if (state == TrainState.IDLE.value):
            return jsonify({'completed': False})
    try:
        redis_client.set(REDIS_KEY_STATE, TrainState.WAIT.value)
        spots = detect_spots(config, redis_client)
        get_mq_connection().channel().basic_publish(
            exchange='',
            routing_key='prediction',
            body='Prediction updated'
        )
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
    if request.headers['Content-Type'] != 'application/json':
        return jsonify(error='Content-Type should be application/json'), 400
    if int(redis_client.get(REDIS_KEY_STATE)) != TrainState.IDLE.value:
        return jsonify(error='Process is running'), 500
    req_json = request.get_json()
    req_json['device'] = device()
    config = ResetConfig(req_json)
    logger().info(config)
    try:
        init_seg_models(config.model_path,
                        config.keep_axials,
                        config.device,
                        config.is_3d,
                        config.n_models,
                        config.n_crops,
                        config.zpath_input,
                        config.crop_size,
                        config.scales,
                        redis_client=redis_client,
                        url=config.url)
    except RuntimeError as e:
        logger().exception('Failed in init_seg_models')
        return jsonify(error=f'Runtime Error: {e}'), 500
    except Exception as e:
        logger().exception('Failed in init_seg_models')
        return jsonify(error=f'Exception: {e}'), 500
    finally:
        gc.collect()
        torch.cuda.empty_cache()
    return jsonify({'completed': True})


@app.route('/export/ctc', methods=['POST'])
def export_ctc():
    if request.headers['Content-Type'] != 'application/json':
        return jsonify(error='Content-Type should be application/json'), 400
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

    state = int(redis_client.get(REDIS_KEY_STATE))
    while (state == TrainState.WAIT.value):
        logger().info("waiting", "@/export/ctc")
        time.sleep(1)
        state = int(redis_client.get(REDIS_KEY_STATE))
        if (state == TrainState.IDLE.value):
            return make_response('', 204)
    try:
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
    for gpu in nvsmi.get_gpus():
        gpus.append({
            'id': gpu.id,
            'name': gpu.name,
            'mem_total': gpu.mem_total,
            'mem_used': gpu.mem_used
        })
    resp = jsonify(gpus)
    return resp


@app.route('/dataset/check', methods=['POST'])
def check_datset():
    if request.headers['Content-Type'] != 'application/json':
        return jsonify(error='Content-Type should be application/json'), 400
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
        return jsonify(error='Content-Type should be application/json'), 400
    req_json = request.get_json()
    if 'dataset_name' not in req_json:
        return jsonify(error='dataset_name key is missing'), 400
    p_dataset = Path(DATASETS_DIR) / req_json['dataset_name']
    h5_files = list(sorted(p_dataset.glob('*.h5')))
    if len(h5_files) == 0:
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
            get_mq_connection(),
        )
    except Exception as e:
        logger().exception('Failed in generate_dataset')
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=80)
