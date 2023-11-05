#! /usr/bin/env python
# ==============================================================================
# Copyright (c) 2023, Ko Sugawara
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
from pathlib import Path

import skimage.io
from tqdm import tqdm

from elephant.common import detect_spots
from elephant.common import export_ctc_labels
from elephant.config import ExportConfig
from elephant.config import SegmentationEvalConfigTiff
from elephant.util import get_next_multiple


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input directory")
    parser.add_argument("config", help="config file")
    parser.add_argument("--output", help="output directory")
    args = parser.parse_args()
    # list up input image files
    files = [
        os.path.join(args.input, f)
        for f in sorted(os.listdir(args.input))
        if f.endswith(".tif")
    ]
    with io.open(args.config, "r", encoding="utf-8") as jsonfile:
        config_data = json.load(jsonfile)
    if not "patch" in config_data:
        is_3d = (
            config_data.get("is_3d", True) == True
            and config_data.get("use_2d", False) == False
        )
        config_data["patch"] = [
            int(get_next_multiple(s * 0.75, 16))
            for s in skimage.io.imread(files[0]).shape[-(2 + is_3d) :]
        ]
    config = SegmentationEvalConfigTiff(config_data)
    print("Start detection...")
    print(config)
    for i, f in tqdm(enumerate(files)):
        config.timepoint = i
        config.tiff_input = f
        spots = detect_spots(
                    str(config.device),
                    config.model_path,
                    config.keep_axials,
                    config.is_pad,
                    config.is_3d,
                    config.crop_size,
                    config.scales,
                    config.cache_maxbytes,
                    config.use_2d,
                    config.use_median,
                    config.patch_size,
                    config.crop_box,
                    config.c_ratio,
                    config.p_thresh,
                    config.r_min,
                    config.r_max,
                    config.output_prediction,
                    None,
                    None,
                    config.timepoint,
                    config.tiff_input,
                    None,
                    config.batch_size,
                    config.input_size,
                )
        print("End detection.")
        print("Start export...")
        Path(args.output).mkdir(parents=True, exist_ok=True)
        config_data["savedir"] = args.output
        config_data["shape"] = skimage.io.imread(f).shape
        config_data["t_start"] = i
        config_data["t_end"] = i
        config_export = ExportConfig(config_data)
        print(config_export)
        # load spots and group by t
        spots_dict = collections.defaultdict(list)
        for spot in spots:
            current_spots = spots_dict.get(spot["t"])
            if current_spots is None:
                spot["value"] = 1
            else:
                spot["value"] = len(current_spots) + 1
            spots_dict[spot["t"]].append(spot)
        spots_dict = collections.OrderedDict(sorted(spots_dict.items()))
        # export labels
        export_ctc_labels(config_export, spots_dict)
        print("End export.")
    print("Done")


if __name__ == "__main__":
    main()