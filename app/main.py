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
import traceback
import weakref

from flask import Flask
from flask import g
from flask import jsonify
from flask import make_response
from flask import request
from flask import send_file
from flask_redis import FlaskRedis
import numpy as np
import pika
import torch
import torch.utils.data as du
import zarr

from elephant.common import TensorBoard
from elephant.common import detect_spots
from elephant.common import export_ctc_labels
from elephant.common import init_flow_models
from elephant.common import init_seg_models
from elephant.common import spots_with_flow
from elephant.common import train
from elephant.config import ExportConfig
from elephant.config import FlowEvalConfig
from elephant.config import FlowTrainConfig
from elephant.config import ResetConfig
from elephant.config import SegmentationEvalConfig
from elephant.config import SegmentationTrainConfig
from elephant.datasets import FlowDatasetZarr
from elephant.datasets import SegmentationDatasetZarr
from elephant.losses import FlowLoss
from elephant.losses import SegmentationLoss
from elephant.models import load_flow_models
from elephant.models import load_seg_models
from elephant.redis_util import REDIS_KEY_COUNT
from elephant.redis_util import REDIS_KEY_LR
from elephant.redis_util import REDIS_KEY_NCROPS
from elephant.redis_util import REDIS_KEY_STATE
from elephant.redis_util import REDIS_KEY_TIMEPOINT
from elephant.redis_util import TrainState
from elephant.util import normalize_zero_one
from elephant.util import get_device
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
        g.connection.channel().queue_declare(queue='prediction')
    if g.connection.is_closed:
        g.connection.open()
    return g.connection


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
            return jsonify(res='error'), 400
        req_json = request.get_json()
        state = req_json.get(REDIS_KEY_STATE)
        if (not isinstance(state, int) or
                state not in TrainState._value2member_map_):
            return jsonify(res='error'), 400
        redis_client.set(REDIS_KEY_STATE, state)
    return jsonify(success=True, state=int(redis_client.get(REDIS_KEY_STATE)))


@app.route('/params', methods=['GET', 'POST'])
def params():
    if request.method == 'POST':
        if request.headers['Content-Type'] != 'application/json':
            return jsonify(res='error'), 400
        req_json = request.get_json()
        lr = req_json.get('lr')
        if not isinstance(lr, float) or lr < 0:
            return jsonify(res='error'), 400
        n_crops = req_json.get('n_crops')
        if not isinstance(n_crops, int) or n_crops < 0:
            return jsonify(res='error'), 400
        redis_client.set(REDIS_KEY_LR, str(lr))
        redis_client.set(REDIS_KEY_NCROPS, str(n_crops))
        print(f'[params updated] lr: {lr}, n_crops: {n_crops}')
    return jsonify(success=True,
                   lr=float(redis_client.get(REDIS_KEY_LR)),
                   n_crops=int(redis_client.get(REDIS_KEY_NCROPS)))


def _update_flow_labels(spots_dict,
                        scales,
                        zpath_flow_label,
                        flow_norm_factor):
    za_label = zarr.open(zpath_flow_label, mode='a')
    n_dims = len(za_label.shape) - 2
    for t, spots in spots_dict.items():
        label = np.zeros(za_label.shape[1:], dtype='float32')
        MIN_AREA_ELLIPSOID = 9
        for spot in spots:
            if int(redis_client.get(REDIS_KEY_STATE)) == TrainState.IDLE.value:
                print('aborted')
                return jsonify({'completed': False})
            centroid = np.array(spot['pos'][::-1])
            covariance = np.array(spot['covariance'][::-1]).reshape(3, 3)
            radii, rotation = np.linalg.eigh(covariance)
            radii = np.sqrt(radii)
            dd, rr, cc = ellipsoid(centroid,
                                   radii,
                                   rotation,
                                   scales,
                                   label.shape[-3:],
                                   MIN_AREA_ELLIPSOID)
            weight = 1  # if spot['tag'] in ['tp'] else false_weight
            displacement = spot['displacement']  # X, Y, Z
            for i in range(n_dims):
                label[i, dd, rr, cc] = displacement[i] / \
                    scales[-1 - i] / flow_norm_factor[i]  # X, Y, Z
            label[-1, dd, rr, cc] = weight  # last channels is for weight
        print('frame:{}, {} linkings'.format(t + 1, len(spots)))
        za_label[t] = label
    return jsonify({'completed': True})


@ app.route('/update/flow', methods=['POST'])
def update_flow_labels():
    if request.headers['Content-Type'] != 'application/json':
        return jsonify(res='error'), 400
    req_json = request.get_json()
    req_json['device'] = device()
    config = FlowTrainConfig(req_json)
    print(config)
    if req_json.get('reset'):
        state = int(redis_client.get(REDIS_KEY_STATE))
        redis_client.set(REDIS_KEY_STATE, TrainState.WAIT.value)
        try:
            zarr.open_like(zarr.open(config.zpath_flow_label, mode='r'),
                           config.zpath_flow_label,
                           mode='w')
        except RuntimeError as e:
            print(traceback.format_exc())
            return jsonify(error=f'Runtime Error: {e}'), 500
        except Exception as e:
            print(traceback.format_exc())
            return jsonify(error=f'Exception: {e}'), 500
        finally:
            redis_client.set(REDIS_KEY_STATE, state)
        return jsonify({'completed': True})

    spots_dict = collections.defaultdict(list)
    for spot in req_json.get('spots'):
        spots_dict[spot['t']].append(spot)
    spots_dict = collections.OrderedDict(sorted(spots_dict.items()))

    state = int(redis_client.get(REDIS_KEY_STATE))
    redis_client.set(REDIS_KEY_STATE, TrainState.WAIT.value)
    try:
        response = _update_flow_labels(spots_dict,
                                       config.scales,
                                       config.zpath_flow_label,
                                       config.flow_norm_factor)
    except RuntimeError as e:
        print(traceback.format_exc())
        return jsonify(error=f'Runtime Error: {e}'), 500
    except Exception as e:
        print(traceback.format_exc())
        return jsonify(error=f'Exception: {e}'), 500
    finally:
        redis_client.set(REDIS_KEY_STATE, state)
    return response


@app.route('/train/flow', methods=['POST'])
def train_flow():
    if request.headers['Content-Type'] != 'application/json':
        return jsonify(res='error'), 400
    if int(redis_client.get(REDIS_KEY_STATE)) == TrainState.RUN.value:
        return jsonify(error='Training process is running'), 500
    req_json = request.get_json()
    req_json['device'] = device()
    config = FlowTrainConfig(req_json)
    print(config)
    input_shape = zarr.open(config.zpath_input, mode='r').shape[-3:]
    spots_dict = collections.defaultdict(list)
    for spot in req_json.get('spots'):
        spots_dict[spot['t']].append(spot)
    spots_dict = collections.OrderedDict(sorted(spots_dict.items()))
    redis_client.set(REDIS_KEY_STATE, TrainState.RUN.value)
    models = load_flow_models(config.model_path,
                              config.keep_axials,
                              config.device)
    try:
        _update_flow_labels(spots_dict,
                            config.scales,
                            config.zpath_flow_label,
                            config.flow_norm_factor)
        train_dataset = FlowDatasetZarr(
            config.zpath_input,
            config.zpath_flow_label,
            list(spots_dict.keys()),
            input_shape,
            config.crop_size,
            config.n_crops,
            scales=config.scales,
            scale_factor_base=config.scale_factor_base,
            rotation_angle=config.rotation_angle,
        )
        if 0 < len(train_dataset):
            loss = FlowLoss().to(config.device)
            optimizers = [torch.optim.Adam(
                model.parameters(), lr=config.lr) for model in models]
            logger = TensorBoard(config.log_dir)
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
                          log_interval=100,
                          tb_logger=logger,
                          redis_client=redis_client)
                    if (int(redis_client.get(REDIS_KEY_STATE)) ==
                            TrainState.IDLE.value):
                        print('training aborted')
                        return jsonify({'completed': False})
                torch.save(models[0].state_dict() if len(models) == 1 else
                           [model.state_dict() for model in models],
                           config.model_path)
    except RuntimeError as e:
        print(traceback.format_exc())
        return jsonify(error=f'Runtime Error: {e}'), 500
    except Exception as e:
        print(traceback.format_exc())
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
        return jsonify(res='error'), 400
    req_json = request.get_json()
    req_json['device'] = device()
    config = FlowEvalConfig(req_json)
    print(config)
    res_spots = []
    if 0 < config.timepoint:
        state = int(redis_client.get(REDIS_KEY_STATE))
        redis_client.set(REDIS_KEY_STATE, TrainState.WAIT.value)
        try:
            spots = req_json.get('spots')
            res_spots = spots_with_flow(config, spots)
        except RuntimeError as e:
            print(traceback.format_exc())
            return jsonify(error=f'Runtime Error: {e}'), 500
        except Exception as e:
            print(traceback.format_exc())
            return jsonify(error=f'Exception: {e}'), 500
        finally:
            gc.collect()
            torch.cuda.empty_cache()
            redis_client.set(REDIS_KEY_STATE, state)
    return jsonify({'spots': res_spots})


@app.route('/reset/flow', methods=['POST'])
def reset_flow_models():
    if request.headers['Content-Type'] != 'application/json':
        return jsonify(res='error'), 400
    req_json = request.get_json()
    req_json['device'] = device()
    config = ResetConfig(req_json)
    print(config)
    try:
        init_flow_models(config)
    except RuntimeError as e:
        print(traceback.format_exc())
        return jsonify(error=f'Runtime Error: {e}'), 500
    except Exception as e:
        print(traceback.format_exc())
        return jsonify(error=f'Exception: {e}'), 500
    return jsonify({'completed': True})


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
                print('aborted')
                return jsonify({'completed': False})
            cnt[spot['tag']] += 1
            centroid = np.array(spot['pos'][::-1])
            covariance = np.array(spot['covariance'][::-1]).reshape(3, 3)
            radii, rotation = np.linalg.eigh(covariance)
            radii = np.sqrt(radii)
            dd_outer, rr_outer, cc_outer = ellipsoid(
                centroid,
                radii,
                rotation,
                scales,
                label.shape,
                MIN_AREA_ELLIPSOID
            )
            label_offset = 0 if spot['tag'] in ['tp', 'tb', 'tn'] else 3
            if spot['tag'] in ('tp', 'fn'):
                dd_inner, rr_inner, cc_inner = ellipsoid(
                    centroid,
                    radii * c_ratio,
                    rotation,
                    scales,
                    label.shape,
                    MIN_AREA_ELLIPSOID
                )
                dd_inner_p, rr_inner_p, cc_inner_p = _dilate_3d_indices(
                    dd_inner, rr_inner, cc_inner, label.shape)
                label[dd_outer, rr_outer, cc_outer] = np.where(
                    np.fmod(label[dd_outer, rr_outer, cc_outer] - 1, 3) <= 1,
                    2 + label_offset,
                    label[dd_outer, rr_outer, cc_outer]
                )
                label[dd_inner_p, rr_inner_p, cc_inner_p] = 2 + label_offset
                label[dd_inner, rr_inner, cc_inner] = np.where(
                    np.fmod(label[dd_inner, rr_inner, cc_inner] - 1, 3) <= 2,
                    3 + label_offset,
                    label[dd_inner, rr_inner, cc_inner]
                )
            elif spot['tag'] in ('tb', 'fb'):
                label[dd_outer, rr_outer, cc_outer] = np.where(
                    np.fmod(label[dd_outer, rr_outer, cc_outer] - 1, 3) <= 1,
                    2 + label_offset,
                    label[dd_outer, rr_outer, cc_outer]
                )
            elif spot['tag'] in ('tn', 'fp'):
                label[dd_outer, rr_outer, cc_outer] = np.where(
                    np.fmod(label[dd_outer, rr_outer, cc_outer] - 1, 3) <= 0,
                    1 + label_offset,
                    label[dd_outer, rr_outer, cc_outer]
                )
        print('frame:{}, {}'.format(
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


@ app.route('/update/seg', methods=['POST'])
def update_seg_labels():
    if request.headers['Content-Type'] != 'application/json':
        return jsonify(res='error'), 400
    req_json = request.get_json()
    req_json['device'] = device()
    config = SegmentationTrainConfig(req_json)
    print(config)
    if req_json.get('reset'):
        state = int(redis_client.get(REDIS_KEY_STATE))
        redis_client.set(REDIS_KEY_STATE, TrainState.WAIT.value)
        try:
            zarr.open_like(zarr.open(config.zpath_seg_label, mode='r'),
                           config.zpath_seg_label,
                           mode='w')
            zarr.open_like(zarr.open(config.zpath_seg_label_vis, mode='r'),
                           config.zpath_seg_label_vis,
                           mode='w')
        except RuntimeError as e:
            print(traceback.format_exc())
            return jsonify(error=f'Runtime Error: {e}'), 500
        except Exception as e:
            print(traceback.format_exc())
            return jsonify(error=f'Exception: {e}'), 500
        finally:
            redis_client.set(REDIS_KEY_STATE, state)
        return jsonify({'completed': True})

    spots_dict = collections.defaultdict(list)
    for spot in req_json.get('spots'):
        spots_dict[spot['t']].append(spot)
    spots_dict = collections.OrderedDict(sorted(spots_dict.items()))

    state = int(redis_client.get(REDIS_KEY_STATE))
    redis_client.set(REDIS_KEY_STATE, TrainState.WAIT.value)
    try:
        response = _update_seg_labels(spots_dict,
                                      config.scales,
                                      config.zpath_input,
                                      config.zpath_seg_label,
                                      config.zpath_seg_label_vis,
                                      config.auto_bg_thresh,
                                      config.c_ratio)
    except RuntimeError as e:
        print(traceback.format_exc())
        return jsonify(error=f'Runtime Error: {e}'), 500
    except Exception as e:
        print(traceback.format_exc())
        return jsonify(error=f'Exception: {e}'), 500
    finally:
        redis_client.set(REDIS_KEY_STATE, state)
    return response


@ app.route('/train/seg', methods=['POST'])
def train_seg():
    if request.headers['Content-Type'] != 'application/json':
        return jsonify(res='error'), 400
    if int(redis_client.get(REDIS_KEY_STATE)) == TrainState.RUN.value:
        return jsonify(error='Training process is running'), 500
    req_json = request.get_json()
    req_json['device'] = device()
    config = SegmentationTrainConfig(req_json)
    print(config)
    spots_dict = collections.defaultdict(list)
    for spot in req_json.get('spots'):
        spots_dict[spot['t']].append(spot)
    spots_dict = collections.OrderedDict(sorted(spots_dict.items()))

    redis_client.set(REDIS_KEY_STATE, TrainState.RUN.value)
    models = load_seg_models(config.model_path,
                             config.keep_axials,
                             config.device)
    try:
        _update_seg_labels(spots_dict,
                           config.scales,
                           config.zpath_input,
                           config.zpath_seg_label,
                           config.zpath_seg_label_vis,
                           config.auto_bg_thresh,
                           config.c_ratio)
        input_shape = zarr.open(config.zpath_input, mode='r').shape[-3:]
        train_dataset = SegmentationDatasetZarr(
            config.zpath_input,
            config.zpath_seg_label,
            list(spots_dict.keys()),
            input_shape,
            config.crop_size,
            config.n_crops,
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
            logger = TensorBoard(config.log_dir)
            loss = SegmentationLoss(class_weights=weight_tensor,
                                    false_weight=config.false_weight)
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
                          log_interval=100,
                          tb_logger=logger,
                          redis_client=redis_client)
                    if (int(redis_client.get(REDIS_KEY_STATE)) ==
                            TrainState.IDLE.value):
                        print('training aborted')
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
        print(traceback.format_exc())
        return jsonify(error=f'Runtime Error: {e}'), 500
    except Exception as e:
        print(traceback.format_exc())
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
        return jsonify(res='error'), 400
    req_json = request.get_json()
    req_json['device'] = device()
    config = SegmentationEvalConfig(req_json)
    print(config)
    state = int(redis_client.get(REDIS_KEY_STATE))
    redis_client.set(REDIS_KEY_STATE, TrainState.WAIT.value)
    try:
        spots = detect_spots(config)
        get_mq_connection().channel().basic_publish(
            exchange='',
            routing_key='prediction',
            body='Prediction updated'
        )
    except RuntimeError as e:
        print(traceback.format_exc())
        return jsonify(error=f'Runtime Error: {e}'), 500
    except Exception as e:
        print(traceback.format_exc())
        return jsonify(error=f'Exception: {e}'), 500
    finally:
        gc.collect()
        torch.cuda.empty_cache()
        redis_client.set(REDIS_KEY_STATE, state)
    return jsonify({'spots': spots})


@app.route('/reset/seg', methods=['POST'])
def reset_seg_models():
    if request.headers['Content-Type'] != 'application/json':
        return jsonify(res='error'), 400
    if int(redis_client.get(REDIS_KEY_STATE)) == TrainState.RUN.value:
        return jsonify(error='Training process is running'), 500
    req_json = request.get_json()
    req_json['device'] = device()
    config = ResetConfig(req_json)
    print(config)
    redis_client.set(REDIS_KEY_STATE, TrainState.RUN.value)
    try:
        init_seg_models(config)
    except RuntimeError as e:
        print(traceback.format_exc())
        return jsonify(error=f'Runtime Error: {e}'), 500
    except Exception as e:
        print(traceback.format_exc())
        return jsonify(error=f'Exception: {e}'), 500
    finally:
        gc.collect()
        torch.cuda.empty_cache()
        redis_client.set(REDIS_KEY_COUNT, 0)
        redis_client.set(REDIS_KEY_STATE, TrainState.IDLE.value)
    return jsonify({'completed': True})


@ app.route('/export/ctc', methods=['POST'])
def export_ctc():
    if request.headers['Content-Type'] != 'application/json':
        return jsonify(res='error'), 400
    req_json = request.get_json()
    req_json['device'] = device()
    config = ExportConfig(req_json)
    za_input = zarr.open(config.zpath_input, mode='r')
    config.shape = za_input.shape[1:]
    print(config)
    spots_dict = collections.defaultdict(list)
    for spot in req_json.get('spots'):
        spots_dict[spot['t']].append(spot)
    spots_dict = collections.OrderedDict(sorted(spots_dict.items()))

    state = int(redis_client.get(REDIS_KEY_STATE))
    redis_client.set(REDIS_KEY_STATE, TrainState.WAIT.value)
    try:
        result = export_ctc_labels(config, spots_dict, redis_client)
        if isinstance(result, str):
            resp = send_file(result)
            file_remover.cleanup_once_done(resp, result)
        elif not result:
            resp = make_response('', 204)
        else:
            resp = make_response('', 200)
    except RuntimeError as e:
        print(traceback.format_exc())
        return jsonify(error=f'Runtime Error: {e}'), 500
    except Exception as e:
        print(traceback.format_exc())
        return jsonify(error=f'Exception: {e}'), 500
    finally:
        redis_client.set(REDIS_KEY_STATE, state)
    return resp


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=80)
