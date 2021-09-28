import numpy as np
import torch

from elephant.util import get_device
from elephant.util import get_next_multiple
from elephant.util import get_pad_size
from elephant.util import normalize_zero_one


def test_get_next_multiple():
    assert get_next_multiple(120, 16) == 128


def test_get_pad_size():
    pad_size = get_pad_size(21, 16)
    assert pad_size == (5, 6)


def test_normalize_zero_one():
    data = np.array(range(5)).astype(float)
    data = normalize_zero_one(data)
    assert data.min() == 0 and data.max() == 1


def test_get_device():
    assert get_device() in (torch.device("cpu"), torch.device("cuda"))
