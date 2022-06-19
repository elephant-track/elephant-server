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
from functools import partial
import json
import multiprocessing as mp
from pathlib import Path
import re

import h5py
import numpy as np
import zarr
from tqdm import tqdm

from elephant.logging import logger
from elephant.logging import publish_mq


def write_chunk(chunk, zpath, t, n_dims, is_2d, input, timepoint, divisor,
                dtype):
    def func(x): return x[0] if is_2d else x
    with h5py.File(input, 'r') as f:
        # https://arxiv.org/pdf/1412.0488.pdf "2.4 HDF5 File Format"
        zarr.open(zpath, 'a')[(t,) + chunk[-n_dims:]] = (
            np.array(func(f[timepoint]['s00']['0']['cells'][chunk])) //
            divisor
        ).astype(dtype)


def generate_dataset(input, output, is_uint16=False, divisor=1., is_2d=False,
                     is_message_queue=False, is_multiprocessing=True):
    """Generate a dataset for ELEPHANT.

    Parameters
    ----------
    input : str
        Input .h5 file.
    output : str or Path
        Output directory.
    is_uint16 : bool
        With this flag, the original image will be stored with uint16.
        If None, determine if uint8 or uint16 dynamically.
        default: False (uint8)
    divisor : float
        Divide the original pixel values by this value.
        (with uint8, the values should be scale-downed to 0-255)
        default: 1.0
    is_2d : bool
        With this flag, the original image will be stored as 2d+time.
        default: False (3d+time)
    is_message_queue : bool
        With this flag, progress is reported using pika.BlockingConnection.
    is_multiprocessing : bool
        With this flag, multiprocessing is enabled.

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
    logger().info(f'input: {input}')
    logger().info(f'output dir: {output}')
    logger().info(f'divisor: {divisor}')
    with h5py.File(input, 'r') as f:
        # timepoints are stored as 't00000', 't000001', ...
        timepoints = list(filter(re.compile(r't\d{5}').search, list(f.keys())))
        def func(x): return x[0] if is_2d else x
        # determine if uint8 or uint16 dynamically
        if is_uint16 is None:
            is_uint16 = False
            for timepoint in tqdm(timepoints):
                if 255 < np.array(
                    func(f[timepoint]['s00']['0']['cells'])
                ).max():
                    is_uint16 = True
                    break
        shape = f[timepoints[0]]['s00']['0']['cells'].shape
    if is_2d:
        shape = shape[-2:]
    n_dims = 3 - is_2d  # 3 or 2
    n_timepoints = len(timepoints)
    p = Path(output)
    chunk_shape = tuple(min(s, 1024) for s in shape[-2:])
    if not is_2d:
        chunk_shape = (1,) + chunk_shape
    zarr.open(
        str(p / 'imgs.zarr'),
        'w',
        shape=(n_timepoints,) + shape,
        chunks=(1,) + chunk_shape,
        dtype='u2' if is_uint16 else 'u1'
    )
    zarr.open(
        str(p / 'flow_outputs.zarr'),
        'w',
        shape=(n_timepoints - 1, n_dims,) + shape,
        chunks=(1, 1,) + chunk_shape,
        dtype='f2'
    )
    zarr.open(
        str(p / 'flow_hashes.zarr'),
        'w', shape=(n_timepoints - 1,),
        chunks=(n_timepoints - 1,),
        dtype='S16'
    )
    zarr.open(
        str(p / 'flow_labels.zarr'),
        'w',
        shape=(n_timepoints - 1, n_dims + 1,) + shape,
        chunks=(1, 1,) + chunk_shape,
        dtype='f4'
    )
    zarr.open(
        str(p / 'seg_outputs.zarr'),
        'w',
        shape=(n_timepoints,) + shape + (3,),
        chunks=(1,) + chunk_shape + (1,),
        dtype='f2'
    )
    zarr.open(
        str(p / 'seg_labels.zarr'),
        'w',
        shape=(n_timepoints,) + shape,
        chunks=(1,) + chunk_shape, dtype='u1'
    )
    zarr.open(
        str(p / 'seg_labels_vis.zarr'),
        'w',
        shape=(n_timepoints,) + shape + (3,),
        chunks=(1,) + chunk_shape + (1,),
        dtype='u1'
    )
    dtype = np.uint16 if is_uint16 else np.uint8
    if n_dims == 2:
        chunks = tuple(
            (slice(None),
                slice(y, y+chunk_shape[-2]),
                slice(x, x+chunk_shape[-1]))
            for y in range(0, shape[-2], chunk_shape[-2])
            for x in range(0, shape[-1], chunk_shape[-1])
        )
    else:
        chunks = tuple(
            (slice(z, z+chunk_shape[-3]),
                slice(y, y+chunk_shape[-2]),
                slice(x, x+chunk_shape[-1]))
            for z in range(0, shape[-3], chunk_shape[-3])
            for y in range(0, shape[-2], chunk_shape[-2])
            for x in range(0, shape[-1], chunk_shape[-1])
        )
    if is_multiprocessing:
        pool = mp.Pool()
    try:
        for t, timepoint in tqdm(enumerate(timepoints)):
            partial_write_chunk = partial(write_chunk,
                                          zpath=str(p / 'imgs.zarr'),
                                          t=t,
                                          n_dims=n_dims,
                                          is_2d=is_2d,
                                          input=input,
                                          timepoint=timepoint,
                                          divisor=divisor,
                                          dtype=dtype)
            if is_multiprocessing:
                pool.map(partial_write_chunk, chunks)
            else:
                for chunk in chunks:
                    partial_write_chunk(chunk)
            if is_message_queue:
                publish_mq('dataset', json.dumps({'t_max': n_timepoints,
                                                  't_current': t + 1, }))
    finally:
        if is_multiprocessing:
            pool.close()


def check_dataset(dataset, shape):
    """Check a dataset for ELEPHANT.

    Parameters
    ----------
    dataset : str or Path
        Dataset dir to check.
    shape : tuple of int
        Expected image shape (timepoints, depth, height, width) for 3D+t data,
        or (timepoints, height, width) for 2D+t data.

    Returns
    ----------
    message : str
        Returns 'ready' if everything is ok, otherwise returns the first
        encountered problem.

    This function checks if the dataset has the foolowing files.
    It also checks if zarr shapes and dtypes are consistent with imgs.zarr.

    dataset
    ├── flow_hashes.zarr
    ├── flow_labels.zarr
    ├── flow_outputs.zarr
    ├── imgs.zarr
    ├── seg_labels_vis.zarr
    ├── seg_labels.zarr
    └── seg_outputs.zarr
    """
    p = Path(dataset)
    message = 'ready'
    try:
        try:
            img = zarr.open(str(p / 'imgs.zarr'), 'r')
        except ValueError:
            raise Exception(f'{p / "imgs.zarr"} is not found or broken.')
        if img.shape != shape:
            raise Exception('Invalid shape for imgs.zarr\n' +
                            f'Expected: {shape} Found: {img.shape}')
        n_dims = len(shape) - 1
        n_timepoints = shape[0]
        _check_zarr(p / 'flow_outputs.zarr',
                    (n_timepoints - 1, n_dims,) + shape[-n_dims:],
                    'float16')
        _check_zarr(p / 'flow_hashes.zarr',
                    (n_timepoints - 1,),
                    'S16')
        _check_zarr(p / 'flow_labels.zarr',
                    (n_timepoints - 1, n_dims + 1,) + shape[-n_dims:],
                    'float32')
        _check_zarr(p / 'seg_outputs.zarr',
                    (n_timepoints,) + shape[-n_dims:] + (3,),
                    'float16')
        _check_zarr(p / 'seg_labels.zarr',
                    (n_timepoints,) + shape[-n_dims:],
                    'uint8')
        _check_zarr(p / 'seg_labels_vis.zarr',
                    (n_timepoints,) + shape[-n_dims:] + (3,),
                    'uint8')
    except Exception as e:
        logger().info(str(e))
        message = str(e)
    return message


def _check_zarr(zarr_path, shape, dtype):
    try:
        za = zarr.open(str(zarr_path), 'r')
    except ValueError:
        raise Exception(f'{zarr_path} is not found or broken.')
    if za.shape != shape:
        raise Exception(f'Invalid shape for {zarr_path.name}\n' +
                        f'Expected: {shape} Found: {za.shape}')
    if za.dtype != dtype:
        raise Exception(f'Invalid dtype for {zarr_path.name}')
