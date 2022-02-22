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

from elephant.logging import logger
from elephant.redis_util import redis_client
from elephant.redis_util import REDIS_KEY_LR
from elephant.redis_util import REDIS_KEY_NCROPS

api = Namespace('params', description='Get state or change params')


@api.route('/')
class Root(Resource):
    @api.doc()
    def get(self):
        '''
        Get current parameters.

        '''
        return make_response(
            jsonify(success=True,
                    lr=float(redis_client.get(REDIS_KEY_LR)),
                    n_crops=int(redis_client.get(REDIS_KEY_NCROPS)))
        )

    def post(self):
        if request.headers['Content-Type'] != 'application/json':
            msg = 'Content-Type should be application/json'
            logger().error(msg)
            return make_response(jsonify(error=msg), 400)
        req_json = request.get_json()
        lr = req_json.get('lr')
        if not isinstance(lr, float) or lr < 0:
            msg = f'Invalid learning rate: {lr}'
            logger().error(msg)
            return make_response(jsonify(error=msg), 400)
        n_crops = req_json.get('n_crops')
        if not isinstance(n_crops, int) or n_crops < 0:
            msg = f'Invalid number of crops: {n_crops}'
            logger().error(msg)
            return make_response(jsonify(error=msg), 400)
        redis_client.set(REDIS_KEY_LR, str(lr))
        redis_client.set(REDIS_KEY_NCROPS, str(n_crops))
        logger().info(f'[params updated] lr: {lr}, n_crops: {n_crops}')
        return make_response(
            jsonify(success=True,
                    lr=float(redis_client.get(REDIS_KEY_LR)),
                    n_crops=int(redis_client.get(REDIS_KEY_NCROPS)))
        )
