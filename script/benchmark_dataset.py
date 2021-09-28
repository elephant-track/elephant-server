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
import argparse
from pathlib import Path
import shutil
import tempfile

import h5py
from multiprocessing import Pool
import numpy as np
import zarr

N_TIMEPOINT = 100
CROP_SHAPE = (16, 384, 384)
IMAGE_SHAPE = (32, 1024, 1024)
H5_IMAGE_KEY = 'image'


def main():
    """Benchmark HDF5 vs Zarr.

    Parameters
    ----------
    backend : str
        `h5` or `zarr`
    nprocs : int
        number of processes

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--backend', default='h5',
                        type=str, help='backend (h5 or zarr)')
    parser.add_argument('-n', '--nprocs', default=1, type=int,
                        help='number of processes for multiprocessing')
    args = parser.parse_args()
    assert args.backend in ('h5', 'zarr')
    p = Path(tempfile.mkdtemp())
    try:
        timepoints = list(range(N_TIMEPOINT))
        if args.backend == 'h5':
            output = str(p / 'output.h5')
            with h5py.File(output, 'w') as fout:
                fout.create_dataset(H5_IMAGE_KEY,
                                    (N_TIMEPOINT,) + IMAGE_SHAPE,
                                    dtype='f',
                                    chunks=(1,) + IMAGE_SHAPE,)
        elif args.backend == 'zarr':
            output = str(p / 'output.zarr')
            zarr.open(output,
                      'w',
                      shape=(N_TIMEPOINT,) + IMAGE_SHAPE,
                      chunks=(1,) + IMAGE_SHAPE,
                      dtype='f')
        try:
            pool = Pool(args.nprocs)
            writer = DataWriter(args.backend, output)
            pool.map(writer, timepoints)
        finally:
            pool.close()
            pool.join()
    finally:
        shutil.rmtree(p)


class DataWriter(object):

    def __init__(self, backend, output):
        self.backend = backend
        self.output = output

    def __call__(self, timepoint):
        start = np.random.randint(
            0,
            np.maximum(1, np.array(IMAGE_SHAPE) - np.array(CROP_SHAPE))
        )
        slices = tuple(slice(s, s+size)
                       for s, size in zip(start, CROP_SHAPE))
        data = np.random.rand(*CROP_SHAPE)
        if self.backend == 'h5':
            ok = False
            while not ok:
                try:
                    with h5py.File(self.output, 'a') as fout:
                        fout[H5_IMAGE_KEY][timepoint][slices] = data
                    ok = True
                except Exception:
                    pass
        elif self.backend == 'zarr':
            zarr.open(self.output, 'a')[timepoint][slices] = data


if __name__ == '__main__':
    main()
