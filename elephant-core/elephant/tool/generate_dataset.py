#! /usr/bin/env python
# ==============================================================================
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
import re
from pathlib import Path

import h5py
import numpy as np
import zarr
from tqdm import tqdm


def generate_dataset(input, output, is_uint16=False, divisor=1., is_2d=False):
    """Generate a dataset for ELEPHANT.

    Parameters
    ----------
    input : str
        Input .h5 file.
    output : str
        Output directory.
    is_uint16 : bool
        With this flag, the original image will be stored with uint16.
        default: False (uint8)
    divisor : float
        Divide the original pixel values by this value.
        (with uint8, the values should be scale-downed to 0-255)
        default: 1.0
    is_2d : bool
        With this flag, the original image will be stored as 2d+time.
        default: False (3d+time)

    This function will generate the following files.

    output
    ├── input.h5
    ├── flow_hashes.zarr
    ├── flow_labels.zarr
    ├── flow_outputs.zarr
    ├── imgs.zarr
    ├── seg_labels_vis.zarr
    ├── seg_labels.zarr
    └── seg_outputs.zarr
    """
    print(f'input: {input}')
    print(f'output dir: {output}')
    print(f'divisor: {divisor}')
    f = h5py.File(input, 'r')
    # timepoints are stored as 't00000', 't000001', ...
    timepoints = list(filter(re.compile(r't\d{5}').search, list(f.keys())))
    shape = f[timepoints[0]]['s00']['0']['cells'].shape
    if is_2d:
        shape = shape[-2:]
    n_dims = 3 - is_2d  # 3 or 2
    p = Path(output)
    img = zarr.open(
        str(p / 'imgs.zarr'),
        'w',
        shape=(len(timepoints),) + shape,
        chunks=(1,) + shape,
        dtype='u2' if is_uint16 else 'u1'
    )
    zarr.open(
        str(p / 'flow_outputs.zarr'),
        'w',
        shape=(len(timepoints) - 1, n_dims,) + shape,
        chunks=(1, 1,) + shape,
        dtype='f2'
    )
    zarr.open(
        str(p / 'flow_hashes.zarr'),
        'w', shape=(len(timepoints) - 1,),
        chunks=(len(timepoints) - 1,),
        dtype='S16'
    )
    zarr.open(
        str(p / 'flow_labels.zarr'),
        'w',
        shape=(len(timepoints) - 1, n_dims + 1,) + shape,
        chunks=(1, 1,) + shape,
        dtype='f4'
    )
    zarr.open(
        str(p / 'seg_outputs.zarr'),
        'w',
        shape=(len(timepoints),) + shape + (3,),
        chunks=(1,) + shape + (3,),
        dtype='f2'
    )
    zarr.open(
        str(p / 'seg_labels.zarr'),
        'w',
        shape=(len(timepoints),) + shape,
        chunks=(1,) + shape, dtype='u1'
    )
    zarr.open(
        str(p / 'seg_labels_vis.zarr'),
        'w',
        shape=(len(timepoints),) + shape + (3,),
        chunks=(1,) + shape + (3,),
        dtype='u1'
    )
    dtype = np.uint16 if is_uint16 else np.uint8
    for timepoint in tqdm(timepoints):
        # https://arxiv.org/pdf/1412.0488.pdf "2.4 HDF5 File Format"
        def func(x): return x[0] if is_2d else x
        img[int(timepoint[1:])] = (
            np.array(func(f[timepoint]['s00']['0']['cells'])) // divisor
        ).astype(dtype)
