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

from elephant.common import detect_spots
from elephant.config import SegmentationEvalConfig


def main():
    config_dict = {
        'dataset_name': "CMU-1",
        'timepoint': 0,
        'model_name': "CMU-1_detection.pth",
        "device": "cuda",
        "is_3d": False,
        "crop_size": [384, 384],
        "scales": [0.5, 0.5],
        "cache_maxbytes": 0,
        "use_median": True,
        "patch": [4096, 4096],
        "use_memmap": False,
    }
    config = SegmentationEvalConfig(config_dict)
    print(config)
    detect_spots(
        str(config.device), config.model_path, config.keep_axials,
        config.is_pad, config.is_3d, config.crop_size, config.scales,
        config.cache_maxbytes, config.use_2d, config.use_median,
        config.patch_size, config.crop_box, config.c_ratio, config.p_thresh,
        config.r_min, config.r_max, config.output_prediction,
        config.zpath_input, config.zpath_seg_output, config.timepoint,
        None, config.memmap_dir, config.batch_size, config.input_size,
    )


if __name__ == '__main__':
    main()
