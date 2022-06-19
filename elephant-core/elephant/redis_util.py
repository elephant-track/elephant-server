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
"""Utils used for Redis."""

from enum import Enum

import redis

from elephant.util import RUN_ON_FLASK

REDIS_KEY_LR = 'lr'
REDIS_KEY_NCROPS = 'n_crops'
REDIS_KEY_STATE = 'state'
REDIS_KEY_TIMEPOINT = 'timepoint'
REDIS_KEY_UPDATE_ONGOING_SEG = 'update_ongoing_seg'
REDIS_KEY_UPDATE_ONGOING_FLOW = 'update_ongoing_flow'


class TrainState(Enum):
    IDLE = 0
    RUN = 1
    WAIT = 2


if RUN_ON_FLASK:
    redis_client = redis.Redis.from_url('redis://localhost:6379/0')
else:
    redis_client = None


def get_state():
    """
    Returns the current state.

    Returns
    -------
    state : int
        current state

    """
    return int(redis_client.get(REDIS_KEY_STATE))
