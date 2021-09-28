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
"""Commandline interface for prediction and export using a config file."""

import argparse
import collections
import io
import json
import os

import skimage.io
from tqdm import tqdm

from elephant.common import detect_spots
from elephant.common import export_ctc_labels
from elephant.common import spots_with_flow
from elephant.config import ExportConfig
from elephant.config import FlowEvalConfigTiff
from elephant.config import SegmentationEvalConfigTiff
from elephant.util import get_next_multiple


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', help='detection | linking | export')
    parser.add_argument('input', help='input directory')
    parser.add_argument('config', help='config file')
    parser.add_argument('spots', help='spots file')
    parser.add_argument('--output', help='output directory')
    args = parser.parse_args()
    # list up input image files
    files = [os.path.join(args.input, f)
             for f in sorted(os.listdir(args.input)) if f.endswith('.tif')]
    with io.open(args.config, 'r', encoding='utf-8') as jsonfile:
        config_data = json.load(jsonfile)
    if args.command == 'detection':
        config_data['patch'] = [int(get_next_multiple(s * 0.75, 16)) for s
                                in skimage.io.imread(files[0]).shape[-2:]]
        config = SegmentationEvalConfigTiff(config_data)
        print(config)
        spots = []
        for i, f in tqdm(enumerate(files)):
            config.timepoint = i
            config.tiff_input = f
            spots.extend(detect_spots(config))
        with open(args.spots, 'w') as f:
            json.dump({'spots': spots}, f)
    elif args.command == 'linking':
        with io.open(args.spots, 'r', encoding='utf-8') as jsonfile:
            spots_config_data = json.load(jsonfile)
            t = spots_config_data.get('t')
            spots = spots_config_data.get('spots')
        config_data['timepoint'] = t
        config_data['tiffinput'] = files[t-1:t+1]
        config = FlowEvalConfigTiff(config_data)
        print(config)
        # estimate previous spot positions with flow
        res_spots = spots_with_flow(config, spots)
        with open(args.spots, 'w') as f:
            json.dump({'spots': res_spots}, f)
    elif args.command == 'export':
        config_data['savedir'] = args.output
        config_data['shape'] = skimage.io.imread(files[0]).shape
        config_data['t_start'] = 0
        config_data['t_end'] = len(files) - 1
        config = ExportConfig(config_data)
        print(config)
        # load spots and group by t
        with io.open(args.spots, 'r', encoding='utf-8') as jsonfile:
            spots_data = json.load(jsonfile)
        spots_dict = collections.defaultdict(list)
        for spot in spots_data:
            spots_dict[spot['t']].append(spot)
        spots_dict = collections.OrderedDict(sorted(spots_dict.items()))
        # export labels
        export_ctc_labels(config, spots_dict)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
