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

from elephant.util import dilate
from elephant.util.ellipse import ellipse
from elephant.util.ellipsoid import ellipsoid
from elephant.util.scaled_moments import radii_and_rotation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', help='root path of CTC dataset')
    parser.add_argument(
        '--sparse',
        help=('specify a json file that contains a dict of file names for '
              'sparse annotations if required')
    )
    parser.add_argument(
        '--3d',
        dest='is_3d',
        action='store_true',
        help=('specify if generate 3D labels (only used for 3D+time datasets)')
    )
    parser.add_argument(
        '--gt',
        dest='gt',
        action='store_true',
        help=('specify if process GT')
    )
    parser.add_argument(
        '--st',
        dest='st',
        action='store_true',
        help=('specify if process ST')
    )
    parser.add_argument(
        '--gtst',
        dest='gtst',
        action='store_true',
        help=('specify if process GT+ST')
    )
    parser.add_argument(
        '--cr',
        type=float,
        default=0.3,
        help=('center ratio')
    )
    parser.add_argument(
        '--minarea',
        type=float,
        default=9.0,
        help=('minimum area to draw')
    )
    args = parser.parse_args()
    ref_types_list = []
    if args.gt:
        ref_types_list.append(('GT',))
    if args.st:
        ref_types_list.append(('ST',))
    if args.gtst:
        ref_types_list.append(('GT', 'ST'))
    if len(ref_types_list) == 0:
        ref_types_list = [('GT',), ('ST',), ('GT', 'ST')]
    if args.sparse is not None:
        with io.open(args.sparse, 'r', encoding='utf-8') as jsonfile:
            sparse_data = json.load(jsonfile)
    else:
        sparse_data = {}
    p = Path(args.dir)
    is_3d = args.is_3d
    seg_dir_name = f'SEG{"_3d" if is_3d else ""}'
    for ref_types in ref_types_list:
        suffix = '+'.join(ref_types)
        for i in range(2):
            unique_files = set()
            shape = None
            dtype = None
            for ref_type in ref_types:
                p_seg = p / f'0{i+1}_{ref_type}' / seg_dir_name
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
                            if is_3d and n_dims == 2:
                                raise ValueError(
                                    '--3d is specified but image is 2D'
                                )
                            print(f'generate {2 + is_3d}D labels')
                            if is_3d:
                                draw_func = ellipsoid
                                dilate_func = dilate.dilate_3d_indices
                            else:
                                draw_func = ellipse
                                dilate_func = dilate.dilate_2d_indices
                            shape = img.shape[-(2 + is_3d):]
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
            p_root = p / f'0{i+1}-{suffix}-seg{"-3d" if is_3d else ""}'
            p_root.mkdir(exist_ok=True)

            chunks = (1, 1) + zarr_shape[-2:] if is_3d else (1,) + zarr_shape[-2:]

            za_img = zarr.open(
                str(p_root / 'imgs.zarr'),
                'w',
                shape=zarr_shape,
                chunks=chunks,
                dtype=zarr_dtype,
            )
            za_seg = zarr.open(
                str(p_root / 'seg_labels.zarr'),
                'w',
                shape=zarr_shape,
                chunks=chunks,
                dtype='u1'
            )
            visited_files = set()
            count = 0
            for ref_type in ref_types:
                p_seg = p / f'0{i+1}_{ref_type}' / seg_dir_name
                if p_seg.exists():
                    last_t = -1
                    for f in tqdm(sorted(p_seg.glob('*.tif'))):
                        if f.name not in visited_files:
                            t = re.findall(r'(\d+)', f.name)[0]
                            # 2d data -> 2d labels or 3d data -> 3d labels
                            if n_dims == 2 or is_3d:
                                img = skimage.io.imread(
                                    str(p / f'0{i+1}' / f't{t}.tif')
                                )
                                if is_3d:
                                    za_img[
                                        count,
                                        :img.shape[-3],
                                        :img.shape[-2],
                                        :img.shape[-1],
                                    ] = img
                                else:
                                    za_img[
                                        count,
                                        :img.shape[-2],
                                        :img.shape[-1],
                                    ] = img
                            # 3d data -> 2d labels
                            else:
                                z = int(re.findall(r'(\d+)', f.name)[1])
                                if t != last_t:
                                    img_cache = skimage.io.imread(
                                        str(p / f'0{i+1}' / f't{t}.tif')
                                    )
                                    last_t = t
                                za_img[
                                    count,
                                    :img_cache.shape[-2],
                                    :img_cache.shape[-1],
                                ] = img_cache[z]
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
                                try:
                                    radii, rotation = radii_and_rotation(
                                        region.moments_central, is_3d
                                    )
                                    if (radii == 0).any():
                                        raise RuntimeError(
                                            'all radii should be positive')
                                except RuntimeError as e:
                                    print(str(e))
                                    continue
                                radii *= 2
                                factor = 1.0
                                while True:
                                    indices_outer = draw_func(
                                        region.centroid[:2 + is_3d],
                                        radii * factor,
                                        rotation,
                                        shape=seg.shape,
                                    )
                                    if len(indices_outer[0]) < args.minarea:
                                        factor *= 1.1
                                    else:
                                        break
                                factor = 1.0
                                while True:
                                    indices_inner = draw_func(
                                        region.centroid[:2 + is_3d],
                                        radii * args.cr * factor,
                                        rotation,
                                        shape=seg.shape,
                                    )
                                    if len(indices_inner[0]) < args.minarea:
                                        factor *= 1.1
                                    else:
                                        break
                                indices_inner_p = dilate_func(
                                    *indices_inner, seg.shape
                                )
                                seg[indices_outer] = np.where(
                                    seg[indices_outer] < 2,
                                    2,
                                    seg[indices_outer]
                                )
                                seg[indices_inner_p] = 2
                                seg[indices_inner] = 3
                            if is_3d:
                                za_seg[
                                    count,
                                    :seg.shape[-3],
                                    :seg.shape[-2],
                                    :seg.shape[-1],
                                ] = seg
                            else:
                                za_seg[
                                    count,
                                    :seg.shape[-2],
                                    :seg.shape[-1],
                                ] = seg
                            count += 1
                            visited_files.add(f.name)


if __name__ == '__main__':
    main()
