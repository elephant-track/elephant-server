import numpy as np


def dilate_2d_indices(rr, cc, shape):
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
    unique_dilate = np.unique(np.stack((rr_dilate, cc_dilate)), axis=1)
    return unique_dilate[0], unique_dilate[1]


def dilate_3d_indices(dd, rr, cc, shape):
    if len(dd) != len(rr) or len(dd) != len(cc):
        raise RuntimeError('indices should have the same length')
    n_pixels = len(dd)
    dd_dilate = np.array([0, ] * (n_pixels * 3 ** 3))
    rr_dilate = np.copy(dd_dilate)
    cc_dilate = np.copy(dd_dilate)
    offset = 0
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                dd_dilate[offset:offset +
                          n_pixels] = (dd + dz).clip(0, shape[0] - 1)
                rr_dilate[offset:offset +
                          n_pixels] = (rr + dy).clip(0, shape[1] - 1)
                cc_dilate[offset:offset +
                          n_pixels] = (cc + dx).clip(0, shape[2] - 1)
                offset += n_pixels
    unique_dilate = np.unique(
        np.stack((dd_dilate, rr_dilate, cc_dilate)), axis=1)
    return unique_dilate[0], unique_dilate[1], unique_dilate[2]
