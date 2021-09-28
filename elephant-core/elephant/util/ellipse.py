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
"""Drawing an ellipse in a scikit-image style.

Reference:
  https://github.com/scikit-image/scikit-image/blob/v0.16.2/skimage/draw/draw.py

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


def ellipse(center, radii, rotation, scales=None, shape=None, minarea=0):
    """Generate coordinates of pixels within ellipsoid.

    Parameters
    ----------
    center : (2,) ndarray of double
        Centre coordinate of ellipse.
    radii : (2,) ndarray of double
        Radii of ellipse
    rotation : (2, 2) ndarray of double
        Rotation matrix of ellipse.
    scales : (2,) array-like of double
        Scales of axes w.r.t. center and radii.
    shape : tuple, optional
        Image shape which is used to determine the maximum extent of output
        pixel coordinates. This is useful for ellipsoids which exceed the
        image size.
        By default the full extent of the ellipse is used.
    minarea : int
        Minumum area to be drawn.
        Ellipse will be enlarged inside the calculated bounding box until it
        reaches to this value.
        This parameter is useful when the user wants to force to draw
        ellipses even when the calculated area is small.

    Returns
    -------
    rr, cc : ndarray of int
        Pixel coordinates of ellipse.
        May be used to directl index into an array, e.g.
        ``img[rr, cc] = 1``
    """
    center = np.array(center)
    radii = np.array(radii)
    rotation = np.array(rotation)
    assert center.shape == (2,)
    assert radii.shape == (2,)
    assert 0 < radii.max(), "radii should contain at least one positive value"
    assert rotation.shape == (2, 2)
    if scales is None:
        scales = (1.,) * 2
    scales = np.array(scales)
    assert scales.shape == (2,)

    scaled_center = center / scales

    # The upper_left_bottom and lower_right_top corners of the smallest cuboid
    # containing the ellipsoid.
    factor = np.array([
        [i, j] for j in (-1, 1) for i in (-1, 1)]).T
    while True:
        radii_rot = np.abs(
            np.diag(1. / scales)@(rotation@(np.diag(radii)@(factor)))
        ).max(axis=1)
        # In the original scikit-image code, ceil and floor were replaced.
        # https://github.com/scikit-image/scikit-image/blob/master/skimage/draw/draw.py#L127
        upper_left = np.floor(scaled_center - radii_rot).astype(int)
        lower_right = np.ceil(scaled_center + radii_rot).astype(int)

        if shape is not None:
            # Constrain upper_left and lower_ight by shape boundary.
            upper_left = np.maximum(
                upper_left, np.array([0, 0]))
            lower_right = np.minimum(
                lower_right, np.array(shape[:2]) - 1)

        bounding_shape = lower_right - upper_left + 1

        r_lim, c_lim = np.ogrid[0:float(bounding_shape[0]),
                                0:float(bounding_shape[1])]
        r_org, c_org = scaled_center - upper_left
        r_rad, c_rad = radii
        rotation_inv = np.linalg.inv(rotation)
        conversion_matrix = rotation_inv.dot(np.diag(scales))
        r, c = (r_lim - r_org), (c_lim - c_org)
        distances = (
            ((r * conversion_matrix[0, 0] +
              c * conversion_matrix[0, 1]) / r_rad) ** 2 +
            ((r * conversion_matrix[1, 0] +
              c * conversion_matrix[1, 1]) / c_rad) ** 2
        )
        if distances.size < minarea:
            old_radii = radii.copy()
            radii *= 1.1
            print('Increase radii from ({}) to ({})'.format(old_radii, radii))
        else:
            break
    distance_thresh = 1
    while True:
        rr, cc = np.nonzero(distances < distance_thresh)
        if len(rr) < minarea:
            distance_thresh *= 1.1
        else:
            break
    rr.flags.writeable = True
    cc.flags.writeable = True
    rr += upper_left[0]
    cc += upper_left[1]
    return rr, cc
