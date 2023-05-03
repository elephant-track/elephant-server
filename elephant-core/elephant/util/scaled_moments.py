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
"""Scaled version of image moments calculation.

Extended from:
  https://github.com/scikit-image/scikit-image/blob/v0.16.2/skimage/measure/_moments.py

  Copyright (C) 2019, the scikit-image team
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
  2. Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the
      distribution.
  3. Neither the name of skimage nor the names of its contributors may be
      used to endorse or promote products derived from this software without
      specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
  IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
  INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
  HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
  STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
  IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
  POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np


def scaled_moments(image, scales=None, order=3):
    """Calculate all raw image moments taken into account the scales up to a
       certain order.
    The following properties can be calculated from raw image moments:
     * Area as: ``M[0, 0]``.
     * Centroid as: {``M[1, 0] / M[0, 0]``, ``M[0, 1] / M[0, 0]``}.
    Note that raw moments are neither translation, scale nor rotation
    invariant.
    Parameters
    ----------
    image : nD double or uint8 array
        Rasterized shape as image.
    scales : array
        Scales with the same size as ndim of the input image.
    order : int, optional
        Maximum order of moments. Default is 3.
    Returns
    -------
    m : (``order + 1``, ``order + 1``) array
        Raw image moments.
    """
    return scaled_moments_central(image,
                                  scales,
                                  (0,) * image.ndim,
                                  order=order)


def scaled_moments_central(image, scales=None, center=None, order=3, **kwargs):
    """Calculate all central image moments taken into account the scales up to
       a certain order.
    The center coordinates (cr, cc) can be calculated from the raw moments as:
    {``M[1, 0] / M[0, 0]``, ``M[0, 1] / M[0, 0]``}.
    Note that central moments are translation invariant but not scale and
    rotation invariant.
    Parameters
    ----------
    image : nD double or uint8 array
        Rasterized shape as image.
    scales : array
        Scales with the same size as ndim of the input image.
    center : tuple of float, optional
        Coordinates of the image centroid. This will be computed if it
        is not provided.
    order : int, optional
        The maximum order of moments computed.
    Returns
    -------
    mu : (``order + 1``, ``order + 1``) array
        Central image moments.
    """
    assert len(scales) == image.ndim
    if center is None:
        center = scaled_centroid(image, scales)
    calc = image.astype(float)
    for (dim, dim_length), scale in zip(enumerate(image.shape), scales):
        delta = np.linspace(0, scale * dim_length, dim_length,
                            endpoint=False, dtype=float) - center[dim]
        powers_of_delta = delta[:, np.newaxis] ** np.arange(order + 1)
        calc = np.rollaxis(calc, dim, image.ndim)
        calc = np.dot(calc, powers_of_delta)
        calc = np.rollaxis(calc, -1, dim)
    return calc


def scaled_centroid(image, scales=None):
    """Return the centroid of an image with scales.
    Parameters
    ----------
    image : array
        The input image.
    scales : array
        Scales with the same size as ndim of the input image.
    Returns
    -------
    center : tuple of float, length ``image.ndim``
        The scaled centroid of the (nonzero) pixels in ``image``.
    """
    if scales is None:
        scales = (1,) * image.ndim
    M = scaled_moments(image, scales, order=1)
    center = (M[tuple(np.eye(image.ndim, dtype=int))]  # array of weighted sums
              # for each axis
              / M[(0,) * image.ndim])  # weighted sum of all points
    return center


def radii_and_rotation(moments_central, is_3d=False):
    if is_3d:
        n_dims = 3
        idx = ((2, 1, 1, 1, 0, 0, 1, 0, 0),
               (0, 1, 0, 1, 2, 1, 0, 1, 0),
               (0, 0, 1, 0, 0, 1, 1, 1, 2))
    else:
        n_dims = 2
        idx = ((2, 1, 1, 0),
               (0, 1, 1, 2))
    cov = moments_central[idx].reshape((n_dims, n_dims))
    if not cov.any():  # if all zeros
        raise RuntimeError('covariance is all zeros')
    cov /= moments_central[(0,) * n_dims]
    eigvals, eigvecs = np.linalg.eigh(cov)
    if ((eigvals < 0).any() or
        np.iscomplex(eigvals).any() or
            np.iscomplex(eigvecs).any()):
        raise RuntimeError('invalid eigen values/vectors')
    radii = np.sqrt(eigvals)
    rotation = eigvecs
    return radii, rotation
