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
"""Test gaussian smoothing function."""

import numpy as np
from numpy.testing import assert_allclose
import torch

from elephant.util.gaussian_smoothing import GaussianSmoothing


def test_gaussian_smoothing_1d():
    gaussian_1d = GaussianSmoothing(1, 3, 2., dim=1, padding=1)
    test = np.ones((1, 1, 5,))
    test = gaussian_1d(torch.from_numpy(test).float())[0, 0].numpy()
    expected = np.array(
        [0.68083227, 1., 1., 1., 0.68083227]).astype(np.float32)
    print(test)
    assert_allclose(test, expected, rtol=1e-05, atol=1e-08)


def test_gaussian_smoothing_2d():
    gaussian_2d = GaussianSmoothing(1, 3, 2., dim=2, padding=1)
    test = np.ones((1, 1, 5, 5,))
    test = gaussian_2d(torch.from_numpy(test).float())[0, 0].numpy()
    expected = np.array([
        [0.4635325, 0.6808322, 0.6808322, 0.6808322, 0.4635325],
        [0.6808322,        1.,        1.,        1., 0.6808322],
        [0.6808322,        1.,        1.,        1., 0.6808322],
        [0.6808322,        1.,        1.,        1., 0.6808322],
        [0.4635325, 0.6808322, 0.6808322, 0.6808322, 0.4635325],
    ]).astype(np.float32)
    assert_allclose(test, expected, rtol=1e-05, atol=1e-08)


def test_gaussian_smoothing_3d():
    gaussian_3d = GaussianSmoothing(1, 3, 2., dim=3, padding=1)
    test = np.ones((1, 1, 5, 5, 5))
    test = gaussian_3d(torch.from_numpy(test).float())[0, 0].numpy()
    expected = np.array([
        [
            [0.3155879, 0.4635326, 0.4635326, 0.4635326, 0.3155879],
            [0.4635326, 0.6808323, 0.6808323, 0.6808323, 0.4635326],
            [0.4635326, 0.6808323, 0.6808323, 0.6808323, 0.4635326],
            [0.4635326, 0.6808323, 0.6808323, 0.6808323, 0.4635326],
            [0.3155879, 0.4635326, 0.4635326, 0.4635326, 0.3155879],
        ],
        [
            [0.4635325, 0.6808322, 0.6808322, 0.6808322, 0.4635325],
            [0.6808322,        1.,        1.,        1., 0.6808322],
            [0.6808322,        1.,        1.,        1., 0.6808322],
            [0.6808322,        1.,        1.,        1., 0.6808322],
            [0.4635325, 0.6808322, 0.6808322, 0.6808322, 0.4635325],
        ],
        [
            [0.4635325, 0.6808322, 0.6808322, 0.6808322, 0.4635325],
            [0.6808322,        1.,        1.,        1., 0.6808322],
            [0.6808322,        1.,        1.,        1., 0.6808322],
            [0.6808322,        1.,        1.,        1., 0.6808322],
            [0.4635325, 0.6808322, 0.6808322, 0.6808322, 0.4635325],
        ],
        [
            [0.4635325, 0.6808322, 0.6808322, 0.6808322, 0.4635325],
            [0.6808322,        1.,        1.,        1., 0.6808322],
            [0.6808322,        1.,        1.,        1., 0.6808322],
            [0.6808322,        1.,        1.,        1., 0.6808322],
            [0.4635325, 0.6808322, 0.6808322, 0.6808322, 0.4635325],
        ],
        [
            [0.3155879, 0.4635326, 0.4635326, 0.4635326, 0.3155879],
            [0.4635326, 0.6808323, 0.6808323, 0.6808323, 0.4635326],
            [0.4635326, 0.6808323, 0.6808323, 0.6808323, 0.4635326],
            [0.4635326, 0.6808323, 0.6808323, 0.6808323, 0.4635326],
            [0.3155879, 0.4635326, 0.4635326, 0.4635326, 0.3155879],
        ]
    ]
    ).astype(np.float32)
    print(test)
    assert_allclose(test, expected, rtol=1e-05, atol=1e-08)
