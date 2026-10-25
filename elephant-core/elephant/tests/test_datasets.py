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
import torch
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


def test_cached_whole_frame_is_normalised_like_an_uncached_read(za, tmp_path):
    """Without img_size, the raw and the normalised file had one name.

    The raw pixels were written first under that name, so the normalised file
    "existed" and the raw frame (0..1000 here) came back as the image, where an
    uncached read gives it normalised to 0..1.
    """
    memmap_dir = tmp_path / 'memmap'
    cached = _get_memmap_or_load(za, 0, memmap_dir=str(memmap_dir),
                                 img_size=None, crop_box=None)
    uncached = _get_memmap_or_load(za, 0, memmap_dir=None,
                                   img_size=None, crop_box=None)
    assert np.allclose(cached, uncached)
    assert float(np.max(cached)) == 1.0


def test_a_cache_file_of_the_old_scheme_is_not_read(za, tmp_path):
    """Files written before the versioned names are ignored, not trusted.

    On the ELEPHANT server a file left by a cropped read (before cropped
    reads stopped using the cache) held a wrong image for Fluo-N3DH-CE 01
    t=0: a trained model gave 38 detections there instead of 4.
    """
    memmap_dir = tmp_path / 'memmap'
    memmap_dir.mkdir()
    for name in ('ds-t0-False.dat', 'ds-t0-False-64-64.dat'):
        stale = np.memmap(memmap_dir / name, dtype='float32', mode='w+',
                          shape=(64, 64))
        stale[:] = 0.5  # not the frame
        stale.flush()
        del stale
    for img_size in (None, (64, 64)):
        img = _get_memmap_or_load(za, 0, memmap_dir=str(memmap_dir),
                                  img_size=img_size, crop_box=None)
        assert np.unravel_index(np.argmax(img), img.shape) == (40, 20)
        assert float(np.max(img)) == 1.0


@pytest.mark.parametrize('run_on_flask', [False, True])
def test_prior_training_items_need_no_label(tmp_path, monkeypatch,
                                            run_on_flask):
    """Prior training (a model reset without pretrained weights) has no label.

    AutoencoderDatasetZarr.__getitem__ left img_label unassigned in its is_ae
    branch, so every item raised "local variable 'img_label' referenced before
    assignment" and POST /seg/reset with url=None failed with HTTP 500. Outside
    Flask, _generate_item also built its crop index pool from the label.
    """
    import elephant.datasets as ds

    monkeypatch.setattr(ds, 'RUN_ON_FLASK', run_on_flask)
    rng = np.random.default_rng(0)
    zpath = tmp_path / 'ds' / 'imgs.zarr'
    arr = zarr.open(str(zpath), mode='w', shape=(3, 64, 64),
                    dtype='float32', chunks=(1, 64, 64))
    arr[:] = rng.random((3, 64, 64), dtype='float32')
    dataset = ds.AutoencoderDatasetZarr(str(zpath), (64, 64), (32, 32), 2,
                                        memmap_dir=str(tmp_path / 'memmap'))
    (x, _), y = dataset[0]
    assert tuple(x.shape) == (1, 32, 32)
    assert torch.equal(x, y)  # the target of the prior is the input itself


def _labelled_dataset(tmp_path, ds):
    """A 3-frame movie whose frames carry a label and an index pool."""
    rng = np.random.default_rng(0)
    zi = tmp_path / 'ds' / 'imgs.zarr'
    zl = tmp_path / 'ds' / 'seg_labels.zarr'
    imgs = zarr.open(str(zi), mode='w', shape=(3, 64, 64),
                     dtype='float32', chunks=(1, 64, 64))
    imgs[:] = rng.random((3, 64, 64), dtype='float32')
    labels = zarr.open(str(zl), mode='w', shape=(3, 64, 64),
                       dtype='uint8', chunks=(1, 64, 64))
    for t in range(3):
        label = np.zeros((64, 64), dtype='uint8')
        centres = rng.integers(8, 56, size=(4, 2))
        for y, x in centres:
            label[y - 3:y + 4, x - 3:x + 4] = 2
            label[y - 1:y + 2, x - 1:x + 2] = 3
        labels[t] = label
        labels.attrs[f'label.indices.{t}'] = centres.tolist()
    labels.attrs['updated'] = True
    return ds.SegmentationDatasetZarr(
        str(zi), str(zl), [0, 1, 2], (64, 64), (32, 32), 4,
        (False,) * 4, cache_maxbytes=1 << 20,
        memmap_dir=str(tmp_path / 'memmap'))


def test_label_attributes_are_read_once_per_frame(tmp_path, monkeypatch):
    """The label attributes are one JSON file holding every frame's index
    pool. Every item opened and parsed it twice; on a 300-frame movie
    (2.5 MB) that was 4/5 of a training step. Outside live mode they cannot
    change during a run, so a dataset reads each frame's pool once and checks
    the 'updated' flag once."""
    import elephant.datasets as ds

    monkeypatch.setattr(ds, 'RUN_ON_FLASK', True)
    dataset = _labelled_dataset(tmp_path, ds)
    opens = []
    real_open = zarr.open

    def counting_open(path, *args, **kwargs):
        if str(path).endswith('seg_labels.zarr'):
            opens.append(path)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(ds.zarr, 'open', counting_open)
    for _ in range(4):
        for i in range(len(dataset)):
            dataset[i]
    assert len(opens) == 3 + 1  # one pool per frame, one flag check
    assert not real_open(dataset.zpath_seg_label, mode='r').attrs['updated']


def test_reading_once_does_not_change_the_items(tmp_path, monkeypatch):
    """The same seeds give the same crops as reading the attributes for
    every item, as the server did before."""
    import random
    import elephant.datasets as ds

    monkeypatch.setattr(ds, 'RUN_ON_FLASK', True)
    dataset = _labelled_dataset(tmp_path, ds)

    def items(read_every_time):
        random.seed(1)
        torch.manual_seed(1)
        out = []
        for i in range(len(dataset)):
            if read_every_time:
                dataset.index_pools.clear()
                dataset.label_update_checked = False
            (x, _), y = dataset[i]
            out.append((x, y))
        return out

    for (x0, y0), (x1, y1) in zip(items(True), items(False)):
        assert torch.equal(x0, x1) and torch.equal(y0, y1)
