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
"""Test scaled moments function."""

import numpy as np
from numpy.testing import assert_allclose
from skimage import data

from elephant.util.scaled_moments import scaled_centroid
from elephant.util.scaled_moments import scaled_moments_central


def test_np_linspace():
    img = data.cell()
    centroid = scaled_centroid(img, scales=(2.0, 1.0))
    scales = (2.0, 1.0)
    expected = {'mean': [48405.98313954717, 8403.084265729778],
                'std': [101553.02372674456, 17625.788452893714]}
    for (dim, dim_length), scale, in zip(enumerate(img.shape), scales):
        delta = np.linspace(0, scale * dim_length, dim_length,
                            endpoint=False, dtype=float) - centroid[dim]
        powers_of_delta = delta[:, np.newaxis] ** np.arange(2 + 1)
        assert powers_of_delta.mean() == expected['mean'][dim]
        assert powers_of_delta.std() == expected['std'][dim]


def test_scaled_centroid():
    img = data.cell()
    centroid = scaled_centroid(img, scales=(2.0, 1.0))
    expected = np.array([663.687213, 274.497211])
    assert_allclose(centroid, expected)


def test_scaled_moments_central():
    img = data.cell()
    mu = scaled_moments_central(img, scales=(2.0, 1.0), order=2)
    expected = np.array([
        [2.46697460e7,   2.38418579e-7,  6.19935538e11],
        [8.94069672e-08, 1.63203515e09,  -1.45016251e12],
        [3.56890541e12,  -5.73021830e12, 9.06283186e16]
    ])
    assert_allclose(mu, expected)
