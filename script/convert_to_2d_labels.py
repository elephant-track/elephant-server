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
from pathlib import Path
import re

import numpy as np
import skimage.io
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', help='root path of CTC dataset')
    parser.add_argument(
        '--gt',
        dest='gt',
        action='store_true',
        help=('specify if GT is converted to 2d in addition to ST')
    )
    args = parser.parse_args()
    p = Path(args.dir)
    MIN_AREA = 9
    ref_types = (('GT',) if args.gt else ()) + ('ST',)
    for ref_type in ref_types:
        for i in range(2):
            p_seg = p / f'0{i+1}_{ref_type}' / 'SEG'
            if p_seg.exists():
                p_seg_org = p_seg.parent / 'SEG_org'
                p_seg.rename(p_seg_org)
                p_seg.mkdir(exist_ok=True)
                for f in tqdm(p_seg_org.glob('*.tif')):
                    t = re.findall(r'(\d+)', f.name)[0]
                    img = skimage.io.imread(str(f))
                    shape = img.shape
                    if len(shape) != 3:
                        raise ValueError(
                            f'len(img.shape) should be 3 but got {len(shape)}'
                        )
                    for z in range(shape[0]):
                        if MIN_AREA <= np.count_nonzero(img[z]):
                            fname = f'man_seg_{t}_{z:03d}.tif'
                            skimage.io.imsave(str(p_seg / fname), img[z])


if __name__ == '__main__':
    main()
