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
from celery import Celery
from flask import Flask
from flask import request
from flask_redis import FlaskRedis

from apis import init_api
from elephant.logging import logger
from elephant.redis_util import TrainState
from elephant.redis_util import REDIS_KEY_STATE


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


init_api(app)


@ app.before_request
def log_before_request():
    if request.endpoint not in (None, 'gpus'):
        logger().info(f'START {request.method} {request.path}')


@ app.after_request
def log_after_request(response):
    if request.endpoint not in (None, 'gpus'):
        logger().info(
            f'DONE {request.method} {request.path} => [{response.status}]'
        )
    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=80)
