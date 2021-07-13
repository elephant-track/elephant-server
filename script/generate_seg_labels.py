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
"""Commandline interface for generating seg labels from CTC GT data."""
import argparse
import io
import json
from pathlib import Path
import re

import numpy as np
import skimage.draw
import skimage.io
import skimage.measure
from tqdm import tqdm
import zarr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', help='root path of CTC dataset')
    parser.add_argument(
        '--sparse',
        help=('specify a json file that contains a dict of file names for '
              'sparse annotations if required')
    )
    args = parser.parse_args()
    if args.sparse is not None:
        with io.open(args.sparse, 'r', encoding='utf-8') as jsonfile:
            sparse_data = json.load(jsonfile)
    else:
        sparse_data = {}
    p = Path(args.dir)
    CR = 0.3
    MIN_AREA = 9
    for ref_types in (('GT',), ('ST',), ('GT', 'ST')):
        suffix = '+'.join(ref_types)
        for i in range(2):
            unique_files = set()
            shape = None
            dtype = None
            for ref_type in ref_types:
                p_seg = p / f'0{i+1}_{ref_type}' / 'SEG'
                if p_seg.exists():
                    for f in p_seg.glob('*.tif'):
                        if shape is None:
                            t = re.findall(r'(\d+)', f.name)[0]
                            img = skimage.io.imread(
                                str(p / f'0{i+1}' / f't{t}.tif')
                            )
                            n_dims = len(img.shape)
                            print(f'found {n_dims}D data')
                            if n_dims not in (2, 3):
                                raise ValueError(
                                    'image dimension should be 2 or 3'
                                )
                            shape = img.shape[-2:]
                            dtype = img.dtype
                        unique_files.add(f.name)
            if len(unique_files) == 0:
                break
            zarr_shape = (len(unique_files),) + shape
            if dtype.type is np.uint8:
                zarr_dtype = 'u1'
            elif dtype.type is np.uint16:
                zarr_dtype = 'u2'
            else:
                raise ValueError(f'dtype {dtype} is not supported')
            print(zarr_shape, zarr_dtype)
            p_root = p / f'0{i+1}-{suffix}-seg'
            p_root.mkdir(exist_ok=True)

            za_img = zarr.open(
                str(p_root / 'imgs.zarr'),
                'w',
                shape=zarr_shape,
                chunks=(1,) + zarr_shape[1:],
                dtype=zarr_dtype,
            )
            za_seg = zarr.open(
                str(p_root / 'seg_labels.zarr'),
                'w',
                shape=zarr_shape,
                chunks=(1,) + zarr_shape[1:],
                dtype='u1'
            )
            visited_files = set()
            count = 0
            for ref_type in ref_types:
                p_seg = p / f'0{i+1}_{ref_type}' / 'SEG'
                if (p_seg).exists():
                    last_t = -1
                    for f in tqdm(sorted(p_seg.glob('*.tif'))):
                        if f.name not in visited_files:
                            t = re.findall(r'(\d+)', f.name)[0]
                            if n_dims == 2:
                                za_img[count] = skimage.io.imread(
                                    str(p / f'0{i+1}' / f't{t}.tif')
                                )
                            else:
                                z = int(re.findall(r'(\d+)', f.name)[1])
                                if t != last_t:
                                    img_cache = skimage.io.imread(
                                        str(p / f'0{i+1}' / f't{t}.tif')
                                    )
                                    last_t = t
                                za_img[count] = img_cache[z]
                            label = skimage.io.imread(str(f))
                            if (ref_type == 'GT' and
                                    f.name in sparse_data.get(f'0{i+1}', [])):
                                seg = np.zeros(label.shape, dtype=np.uint8)
                            else:
                                seg = np.ones(label.shape, dtype=np.uint8)
                            regions = skimage.measure.regionprops(label)
                            for region in regions:
                                if region.minor_axis_length == 0:
                                    continue
                                factor = 0.5
                                while True:
                                    indices_outer = skimage.draw.ellipse(
                                        region.centroid[0],
                                        region.centroid[1],
                                        region.minor_axis_length * factor,
                                        region.major_axis_length * factor,
                                        seg.shape,
                                        region.orientation + np.pi / 2
                                    )
                                    if len(indices_outer[0]) < MIN_AREA:
                                        factor *= 1.1
                                    else:
                                        break
                                factor = 0.5
                                while True:
                                    indices_inner = skimage.draw.ellipse(
                                        region.centroid[0],
                                        region.centroid[1],
                                        region.minor_axis_length * CR * factor,
                                        region.major_axis_length * CR * factor,
                                        seg.shape,
                                        region.orientation + np.pi / 2
                                    )
                                    if len(indices_inner[0]) < MIN_AREA:
                                        factor *= 1.1
                                    else:
                                        break
                                indices_inner_p = _dilate_2d_indices(
                                    *indices_inner, seg.shape
                                )
                                seg[indices_outer] = np.where(
                                    seg[indices_outer] < 2,
                                    2,
                                    seg[indices_outer]
                                )
                                seg[indices_inner_p] = 2
                                seg[indices_inner] = 3
                            za_seg[count] = seg
                            count += 1
                            visited_files.add(f.name)


def _dilate_2d_indices(rr, cc, shape):
    if len(rr) != len(cc):
        raise RuntimeError('indices should have the same length')
    n_pixels = len(rr)
    rr_dilate = np.array([0, ] * (n_pixels * 3 ** 2))
    cc_dilate = np.copy(rr_dilate)
    offset = 0
    try:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                rr_dilate[offset:offset +
                          n_pixels] = (rr + dy).clip(0, shape[0] - 1)
                cc_dilate[offset:offset +
                          n_pixels] = (cc + dx).clip(0, shape[1] - 1)
                offset += n_pixels
    except IndexError:
        print(rr, cc, shape)
    unique_dilate = np.unique(
        np.stack((rr_dilate, cc_dilate)), axis=1)
    return unique_dilate[0], unique_dilate[1]


if __name__ == '__main__':
    main()
