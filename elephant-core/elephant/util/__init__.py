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

import torch


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
    if data.min() == data.max():
        return data
    data -= data.min()
    data /= data.max()
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
