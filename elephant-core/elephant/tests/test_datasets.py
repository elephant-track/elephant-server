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
"""Test datasets."""

import numpy as np
import pytest
import zarr

from elephant.datasets import _get_memmap_or_load


CROP = [0, 32, 16, 1, 16, 16]  # [z, y, x, d, h, w]


@pytest.fixture
def za(tmp_path):
    """A 2D zarr with a single bright pixel inside the crop region."""
    frame = np.zeros((64, 64), dtype='float32')
    frame[40, 20] = 1000.0  # inside CROP, at (8, 4) of the crop
    zpath = tmp_path / 'ds' / 'imgs.zarr'
    arr = zarr.open(str(zpath), mode='w', shape=(2, 64, 64),
                    dtype='float32', chunks=(1, 64, 64))
    arr[0] = frame
    arr[1] = frame
    return arr


def test_cropped_read_returns_the_crop(za, tmp_path):
    """The cropped return used to slice the cached crop by frame offsets.

    `slices` are absolute offsets into the full frame, but the cached file
    holds only the crop, so a crop at y=32 of a 64-row frame indexed rows
    32.. of a 16-row array and came back empty.
    """
    memmap_dir = tmp_path / 'memmap'
    img = _get_memmap_or_load(za, 0, memmap_dir=str(memmap_dir),
                              img_size=(16, 16), crop_box=list(CROP))
    assert img.shape == (16, 16)
    # The bright pixel is where the crop puts it, so this is the right region.
    assert np.unravel_index(np.argmax(img), img.shape) == (8, 4)


def test_cropped_read_without_img_size_returns_the_crop(za, tmp_path):
    memmap_dir = tmp_path / 'memmap'
    img = _get_memmap_or_load(za, 0, memmap_dir=str(memmap_dir),
                              img_size=None, crop_box=list(CROP))
    assert img.shape == (16, 16)
    assert np.unravel_index(np.argmax(img), img.shape) == (8, 4)


def test_cropped_read_does_not_poison_the_timepoint(za, tmp_path):
    """A cropped read must not write the timepoint's whole-frame cache.

    fpath_org's key is '{dataset}-t{timepoint}-{use_median}' with no crop in
    it, but it was created with the CROP's shape and content. Every later read
    of that timepoint then opened a file named "the whole frame at t" holding
    only some crop, and raised "mmap length is greater than file size" -- the
    HTTP 500s seen on every predict after a cropped one, persisting on disk
    until the .dat files were removed.
    """
    memmap_dir = tmp_path / 'memmap'
    _get_memmap_or_load(za, 0, memmap_dir=str(memmap_dir),
                        img_size=(16, 16), crop_box=list(CROP))
    assert list(memmap_dir.glob('*.dat')) == []

    full = _get_memmap_or_load(za, 0, memmap_dir=str(memmap_dir),
                               img_size=None, crop_box=None)
    assert full.shape == (64, 64)
    assert np.unravel_index(np.argmax(full), full.shape) == (40, 20)


def test_whole_frame_reads_are_still_cached(za, tmp_path):
    """The fix switches the cache off for crops only."""
    memmap_dir = tmp_path / 'memmap'
    first = _get_memmap_or_load(za, 1, memmap_dir=str(memmap_dir),
                                img_size=None, crop_box=None)
    assert list(memmap_dir.glob('*.dat'))
    second = _get_memmap_or_load(za, 1, memmap_dir=str(memmap_dir),
                                 img_size=None, crop_box=None)
    assert np.array_equal(first, second)
