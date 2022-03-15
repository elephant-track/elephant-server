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
import time
import os
from pathlib import Path
import weakref

from flask import jsonify
from flask import make_response
from flask import request
from flask import send_file
from flask_restx import Namespace
from flask_restx import Resource
import zarr

from elephant.common import export_ctc_labels
from elephant.config import BaseConfig
from elephant.config import ExportConfig
from elephant.logging import logger
from elephant.redis_util import get_state
from elephant.redis_util import redis_client
from elephant.redis_util import TrainState
from elephant.redis_util import REDIS_KEY_STATE
from elephant.util import get_device

api = Namespace('download', description='Download APIs')


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


@api.route('/ctc')
class CTC(Resource):
    @api.doc()
    def post(self):
        '''
        Download outputs in CTC format.

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
                return make_response('', 204)
        try:
            req_json = request.get_json()
            req_json['device'] = get_device()
            config = ExportConfig(req_json)
            za_input = zarr.open(config.zpath_input, mode='r')
            config.shape = za_input.shape[1:]
            logger().info(config)
            spots_dict = collections.defaultdict(list)
            for spot in req_json.get('spots'):
                spots_dict[spot['t']].append(spot)
            spots_dict = collections.OrderedDict(sorted(spots_dict.items()))

            redis_client.set(REDIS_KEY_STATE, TrainState.WAIT.value)
            result = export_ctc_labels(config, spots_dict)
            if isinstance(result, str):
                resp = send_file(result)
                # file_remover.cleanup_once_done(resp, result)
            elif not result:
                resp = make_response('', 204)
            else:
                resp = make_response('', 200)
        except RuntimeError as e:
            logger().exception('Failed in export_ctc_labels')
            return make_response(jsonify(error=f'Runtime Error: {e}'), 500)
        except Exception as e:
            logger().exception('Failed in export_ctc_labels')
            return make_response(jsonify(error=f'Exception: {e}'), 500)
        finally:
            if get_state() != TrainState.IDLE.value:
                redis_client.set(REDIS_KEY_STATE, state)
        return resp


@ api.route('/model')
class Model(Resource):
    @ api.doc()
    def post(self):
        '''
        Download a model paramter file.

        '''
        if request.headers['Content-Type'] != 'application/json':
            msg = 'Content-Type should be application/json'
            logger().error(msg)
            return make_response(jsonify(error=msg), 400)
        req_json = request.get_json()
        config = BaseConfig(req_json)
        logger().info(config)
        if not Path(config.model_path).exists():
            logger().info(
                f'model file {config.model_path} not found @{request.path}')
            return make_response(
                jsonify(message=f'model file {config.model_path} not found'),
                204
            )
        try:
            resp = send_file(config.model_path)
        except Exception as e:
            logger().exception(
                'Failed to prepare a model parameter file for download')
            return make_response(jsonify(error=f'Exception: {e}'), 500)
        return resp
