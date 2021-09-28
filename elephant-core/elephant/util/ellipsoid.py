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
"""Drawing an ellipsoid in a scikit-image style.

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


def ellipsoid(center, radii, rotation, scales=None, shape=None, minarea=0):
    """Generate coordinates of pixels within ellipsoid.

    Parameters
    ----------
    center : (3,) ndarray of double
        Centre coordinate of ellipsoid.
    radii : (3,) ndarray of double
        Radii of ellipsoid
    rotation : (3, 3) ndarray of double
        Rotation matrix of ellipsoid.
    scales : (3,) array-like of double
        Scales of axes w.r.t. center and radii.
    shape : tuple, optional
        Image shape which is used to determine the maximum extent of output
        pixel coordinates. This is useful for ellipsoids which exceed the
        image size.
        By default the full extent of the ellipsoid is used.
    minarea : int
        Minumum area to be drawn.
        Ellipsoid will be enlarged inside the calculated bounding box until it
        reaches to this value.
        This parameter is useful when the user wants to force to draw
        ellipsoids even when the calculated area is small.

    Returns
    -------
    dd, rr, cc : ndarray of int
        Voxel coordinates of ellipsoid.
        May be used to directl index into an array, e.g.
        ``img[dd, rr, cc] = 1``
    """
    center = np.array(center)
    radii = np.array(radii)
    rotation = np.array(rotation)
    assert center.shape == (3,)
    assert radii.shape == (3,)
    assert 0 < radii.max(), "radii should contain at least one positive value"
    assert rotation.shape == (3, 3)
    if scales is None:
        scales = (1.,) * 3
    scales = np.array(scales)
    assert scales.shape == (3,)

    scaled_center = center / scales

    # The upper_left_bottom and lower_right_top corners of the smallest cuboid
    # containing the ellipsoid.
    factor = np.array([
        [i, j, k] for k in (-1, 1) for j in (-1, 1) for i in (-1, 1)]).T
    while True:
        radii_rot = np.abs(
            np.diag(1. / scales).dot(rotation.dot(np.diag(radii).dot(factor)))
        ).max(axis=1)
        # In the original scikit-image code, ceil and floor were replaced.
        # https://github.com/scikit-image/scikit-image/blob/master/skimage/draw/draw.py#L127
        upper_left_bottom = np.floor(scaled_center - radii_rot).astype(int)
        lower_right_top = np.ceil(scaled_center + radii_rot).astype(int)

        if shape is not None:
            # Constrain upper_left and lower_ight by shape boundary.
            upper_left_bottom = np.maximum(
                upper_left_bottom, np.array([0, 0, 0]))
            lower_right_top = np.minimum(
                lower_right_top, np.array(shape[:3]) - 1)

        bounding_shape = lower_right_top - upper_left_bottom + 1

        d_lim, r_lim, c_lim = np.ogrid[0:float(bounding_shape[0]),
                                       0:float(bounding_shape[1]),
                                       0:float(bounding_shape[2])]
        d_org, r_org, c_org = scaled_center - upper_left_bottom
        d_rad, r_rad, c_rad = radii
        rotation_inv = np.linalg.inv(rotation)
        conversion_matrix = rotation_inv.dot(np.diag(scales))
        d, r, c = (d_lim - d_org), (r_lim - r_org), (c_lim - c_org)
        distances = (
            ((d * conversion_matrix[0, 0] +
              r * conversion_matrix[0, 1] +
              c * conversion_matrix[0, 2]) / d_rad) ** 2 +
            ((d * conversion_matrix[1, 0] +
              r * conversion_matrix[1, 1] +
              c * conversion_matrix[1, 2]) / r_rad) ** 2 +
            ((d * conversion_matrix[2, 0] +
              r * conversion_matrix[2, 1] +
              c * conversion_matrix[2, 2]) / c_rad) ** 2
        )
        if distances.size < minarea:
            old_radii = radii.copy()
            radii *= 1.1
            print('Increase radii from ({}) to ({})'.format(old_radii, radii))
        else:
            break
    distance_thresh = 1
    while True:
        dd, rr, cc = np.nonzero(distances < distance_thresh)
        if len(dd) < minarea:
            distance_thresh *= 1.1
        else:
            break
    dd.flags.writeable = True
    rr.flags.writeable = True
    cc.flags.writeable = True
    dd += upper_left_bottom[0]
    rr += upper_left_bottom[1]
    cc += upper_left_bottom[2]
    return dd, rr, cc
