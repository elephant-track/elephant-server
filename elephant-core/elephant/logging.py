# Copyright (c) 2021, Ko Sugawara
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
"""Logger implementation for RabbitMQ."""
import logging
import json
import os

import pika

logging.getLogger("pika").setLevel(logging.WARNING)

RUN_ON_FLASK = "RUN_ON_FLASK" in os.environ


class RabbitMQHandler(logging.StreamHandler):
    """
    A handler class that delegate logs to RabbitMQ.
    """

    def __init__(self):
        super().__init__()

    def emit(self, record):
        try:
            msg = self.format(record)
            publish_mq('log', json.dumps({'level': record.levelname,
                                          'message': msg, }))
            stream = self.stream
            # issue 35046: merged two stream.writes into one.
            stream.write(msg + self.terminator)
            self.flush()
        except RecursionError:  # See issue 36272
            raise
        except Exception:
            self.handleError(record)


def logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if (all(not isinstance(h, RabbitMQHandler) for h in logger.handlers)):
        logger.addHandler(RabbitMQHandler())
    return logger


def publish_mq(queue, body):
    if RUN_ON_FLASK:
        with pika.BlockingConnection(pika.ConnectionParameters(
                host='localhost', heartbeat=0)) as connection:
            connection.channel().queue_declare(queue=queue)
            connection.channel().basic_publish(exchange='',
                                               routing_key=queue,
                                               body=body)
