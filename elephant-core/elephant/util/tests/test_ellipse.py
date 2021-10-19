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
"""Test ellipse function."""

import numpy as np
from numpy.testing import assert_array_equal

from elephant.util.ellipse import ellipse


def test_ellipse():
    test = np.zeros((5, 5))
    rr, cc = ellipse((2., 2.), (1., 2.), ((1, 0), (0, 1)))
    test[rr, cc] = 1
    expected = np.array([[0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 1, 1, 1, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0]])
    assert_array_equal(test, expected)

    # Rotate 45 degree counterclockwise
    test_rotate = np.zeros((5, 5))
    theta = np.pi / 4
    rotation = (
        (np.cos(theta), np.sin(theta)),
        (-np.sin(theta), np.cos(theta))
    )
    rr, cc = ellipse((2., 2.), (1., 2.), rotation)
    test_rotate[rr, cc] = 1
    expected_rotate = np.array([[0, 0, 0, 0, 0],
                                [0, 1, 1, 0, 0],
                                [0, 1, 1, 1, 0],
                                [0, 0, 1, 1, 0],
                                [0, 0, 0, 0, 0]])
    assert_array_equal(test_rotate, expected_rotate)

    # Change scale in y axis
    test_scale = np.zeros((5, 5))
    rr, cc = ellipse((2., 2.), (1., 2.), ((1, 0), (0, 1)),
                     scales=(0.7, 1), shape=(5, 5))
    test_scale[rr, cc] = 1
    expected_scale = np.array([[0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0],
                               [0, 1, 1, 1, 0],
                               [0, 1, 1, 1, 0],
                               [0, 1, 1, 1, 0]])
    assert_array_equal(test_scale, expected_scale)

    # Specify minarea
    test_minarea = np.zeros((5, 5))
    rr, cc = ellipse((2., 2.), (1., 2.), ((1, 0), (0, 1)), minarea=5)
    test_minarea[rr, cc] = 1
    expected_minarea = np.array([[0, 0, 0, 0, 0],
                                 [0, 0, 1, 0, 0],
                                 [1, 1, 1, 1, 1],
                                 [0, 0, 1, 0, 0],
                                 [0, 0, 0, 0, 0]])
    assert_array_equal(test_minarea, expected_minarea)
