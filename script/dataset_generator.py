#! /usr/bin/env python
# ==============================================================================
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
"""A script for generating .zarr files for the ELEPHANT server."""

import argparse

from elephant.tool.dataset import generate_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input',
        help='input .h5 file'
    )
    parser.add_argument(
        'output',
        help='output directory'
    )
    parser.add_argument(
        '--uint16',
        dest='is_uint16',
        action='store_true',
        help=('with this flag, the original image will be stored with uint16' +
              'default: False (uint8)')
    )
    parser.add_argument(
        '--divisor',
        type=float,
        default=1.,
        help=('divide the original pixel values by this value ' +
              '(with uint8, the values should be scale-downed to 0-255)')
    )
    parser.add_argument(
        '--2d',
        dest='is_2d',
        action='store_true',
        help=('with this flag, the original image will be stored as 2d+time' +
              'default: False (3d+time)')
    )
    return parser.parse_args()


def main():
    args = parse_args()
    generate_dataset(
        args.input,
        args.output,
        args.is_uint16,
        args.divisor,
        args.is_2d,
    )


if __name__ == '__main__':
    main()
