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
from flask import jsonify
from flask import make_response
from flask import request
from flask_restx import Namespace
from flask_restx import Resource
import nvsmi
import psutil


from elephant.logging import logger
from elephant.redis_util import get_state
from elephant.redis_util import redis_client
from elephant.redis_util import REDIS_KEY_STATE
from elephant.redis_util import TrainState

api = Namespace('state', description='Get or update state')


@api.route('/process')
class Process(Resource):
    @api.doc()
    def get(self):
        '''
        Check Process state.

        '''
        return make_response(
            jsonify(success=True, state=get_state())
        )

    def post(self):
        '''
        Update process state.

        '''
        if request.headers['Content-Type'] != 'application/json':
            msg = 'Content-Type should be application/json'
            logger().error(msg)
            return make_response(jsonify(error=msg), 400)
        req_json = request.get_json()
        state = req_json.get(REDIS_KEY_STATE)
        if (not isinstance(state, int) or
                state not in TrainState._value2member_map_):
            msg = f'Invalid state: {state}'
            logger().error(msg)
            return make_response(jsonify(res=msg), 400)
        redis_client.set(REDIS_KEY_STATE, state)
        return make_response(
            jsonify(success=True, state=get_state())
        )


@ api.route('/gpus', endpoint='gpus')
class GPUs(Resource):
    @ api.doc()
    def get(self):
        '''
        Get GPU status.

        '''
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
