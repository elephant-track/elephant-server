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
"""Exposing utility functions."""

from collections import OrderedDict
import os
import sys

import numpy as np
import torch

RUN_ON_FLASK = "RUN_ON_FLASK" in os.environ


def to_fancy_index(*data):
    """Build fancy index with broadcast.

    Parameters
    ----------
    *data : int, ndarray
        Indices to be used for building fancy index.

    Example
    -------
    >>> to_fancy_index(0, np.array([1, 1]), np.array([2, 2]), 3)
    (array([0, 0]), array([1, 1]), array([2, 2]), array([3, 3]))

    """
    return tuple(np.array(tuple(np.broadcast(*data))).T)


def get_next_multiple(value, base):
    """Calculate the next multiple of a number.

    Args:
        value (int): a number to be evaluated.
        base (int): a base number for this calculation.
    """
    return ((value - 1) // base + 1) * base


def get_pad_size(size, base):
    assert float(size).is_integer(), 'size should be integer value'
    pad_total = get_next_multiple(size, base) - size
    return pad_total // 2, pad_total - (pad_total // 2)


def normalize_zero_one(data):
    data -= data.min()
    data_max = data.max()
    if 0 < data_max:
        data /= data_max
    return data


def get_device():
    """check if we have a gpu"""
    if torch.cuda.is_available():
        print("GPU is available")
        device = torch.device("cuda")
    else:
        print("GPU is not available")
        device = torch.device("cpu")
    return device


class LRUCacheDict:

    def __init__(self, maxbytes=1024*1024):
        """ Construct a LRUCacheDict with the cache size.

        Args:
            maxbytes (int): size of the memory capacity for cache in byte.
        """
        self.cache = OrderedDict()
        self.maxbytes = maxbytes

    def clear(self):
        self.cache = OrderedDict()

    def get(self, key, default=None):
        if key not in self.cache:
            if default is None:
                return None
            self.cache[key] = default
        self.cache.move_to_end(key)
        item = self.cache[key]
        while self.maxbytes < self.size():
            self.cache.popitem(0)
        return item

    def size(self):
        return sum(sys.getsizeof(v) for v in self.cache.values())
