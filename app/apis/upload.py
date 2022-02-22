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
import os
from pathlib import Path
import tempfile

from elephant.config import DATASETS_DIR
from flask import jsonify
from flask import make_response
from flask import request
from flask_restx import Namespace
from flask_restx import Resource
import werkzeug

api = Namespace('upload', description='Upload APIs')


@api.route('/image')
class Image(Resource):
    @api.doc()
    def post(self):
        """
        Upload an image data to the dataset directory. # noqa: E501

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
                return make_response(
                    jsonify(res='dataset_name is not specified'), 400
                )
            if 'filename' not in form:
                return make_response(
                    jsonify(res='filename is not specified'), 400
                )
            if 'action' not in form:
                return make_response(
                    jsonify(res='action is not specified'), 400
                )
            if form['action'] not in ('init', 'append', 'complete', 'cancel'):
                return make_response(
                    jsonify(res=f'unknown action: {form["action"]}'), 400
                )
            p_file = (Path(DATASETS_DIR) /
                      form['dataset_name'] / form['filename'])
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
