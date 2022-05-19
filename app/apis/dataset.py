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
from pathlib import Path

from celery import shared_task
from flask import jsonify
from flask import make_response
from flask import request
from flask_restx import Namespace
from flask_restx import Resource

from elephant.config import DATASETS_DIR
from elephant.logging import logger
from elephant.tool import dataset as dstool

api = Namespace('dataset', description='Dataset APIs')


@shared_task()
def generate_dataset_task(input, output, is_uint16=False, divisor=1.,
                          is_2d=False):
    dstool.generate_dataset(input,
                            output,
                            is_uint16,
                            divisor,
                            is_2d,
                            True)


@api.route('/check')
class Check(Resource):
    @api.doc()
    def post(self):
        '''
        Validate Dataset.

        '''
        if request.headers['Content-Type'] != 'application/json':
            msg = 'Content-Type should be application/json'
            logger().error(msg)
            return make_response(jsonify(error=msg), 400)
        req_json = request.get_json()
        if 'dataset_name' not in req_json:
            return make_response(
                jsonify(error='dataset_name key is missing'),
                400
            )
        if 'shape' not in req_json:
            return make_response(jsonify(error='shape key is missing'), 400)
        message = dstool.check_dataset(
            Path(DATASETS_DIR) / req_json['dataset_name'],
            tuple(req_json['shape'])
        )
        return make_response(jsonify(message=message), 200)


@api.route('/generate')
class Generate(Resource):
    @api.doc()
    def post(self):
        '''
        Generate Dataset.

        '''
        if request.headers['Content-Type'] != 'application/json':
            msg = 'Content-Type should be application/json'
            logger().error(msg)
            return make_response(jsonify(error=msg), 400)
        req_json = request.get_json()
        if 'dataset_name' not in req_json:
            return make_response(
                jsonify(error='dataset_name key is missing'),
                400
            )
        p_dataset = Path(DATASETS_DIR) / req_json['dataset_name']
        h5_files = list(sorted(p_dataset.glob('*.h5')))
        if len(h5_files) == 0:
            logger().info(f'.h5 file not found @{request.path}')
            return make_response(
                jsonify(
                    message=f'.h5 file not found in {req_json["dataset_name"]}'
                ),
                204
            )
        if p_dataset / (p_dataset.name + '.h5') in h5_files:
            h5_filename = str(p_dataset / (p_dataset.name + '.h5'))
        else:
            h5_filename = str(h5_files[0])
        logger().info(f'multiple .h5 files found, use {h5_filename}')
        try:
            generate_dataset_task.delay(
                h5_filename,
                str(p_dataset),
                req_json.get('is_uint16', None),
                req_json.get('divisor', 1.),
                req_json.get('is_2d', False),
            ).wait()
        except Exception as e:
            logger().exception('Failed in gen_datset')
            return make_response(jsonify(error=f'Exception: {e}'), 500)
        return make_response('', 200)
