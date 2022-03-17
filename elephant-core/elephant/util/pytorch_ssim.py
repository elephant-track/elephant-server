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
"""3D-compatible SSIM calculation.

Extended from:
  https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py
  (MIT LICENSE)
"""

from math import exp

import torch
import torch.nn.functional as F
from torch.autograd import Variable


def gaussian(window_size, sigma):
    gauss = torch.Tensor([
        exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
        for x in range(window_size)
    ])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = (
        _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0))
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1,
                         window,
                         padding=window_size // 2,
                         groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2,
                         window,
                         padding=window_size // 2,
                         groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2,
                       window,
                       padding=window_size // 2,
                       groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = (
        ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) /
        ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if (channel == self.channel and
                self.window.data.type() == img1.data.type()):
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(
            img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def create_window3d(window_size, channel):
    _z_window = gaussian(window_size[0], 1.5).unsqueeze(1)
    _y_window = gaussian(window_size[1], 1.5).unsqueeze(0)
    _x_window = gaussian(window_size[2], 1.5).unsqueeze(0)
    _3D_window = (
        _z_window.mm(_y_window).unsqueeze(2)
        .matmul(_z_window.mm(_x_window).unsqueeze(1))
        .float().unsqueeze(0).unsqueeze(0)
    )
    window = Variable(
        _3D_window.expand(
            channel,
            1,
            window_size[0],
            window_size[1],
            window_size[2]
        ).contiguous()
    )
    return window / window.sum()


def _ssim3d(img1, img2, window, window_size, channel,
            size_average=True, ignore_zeros=True):
    padding = tuple(w // 2 for w in window_size)
    mu1 = F.conv3d(img1, window, padding=padding, groups=channel)
    mu2 = F.conv3d(img2, window, padding=padding, groups=channel)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(img1 * img1,
                         window,
                         padding=padding,
                         groups=channel) - mu1_sq
    sigma2_sq = F.conv3d(img2 * img2,
                         window,
                         padding=padding,
                         groups=channel) - mu2_sq
    sigma12 = F.conv3d(img1 * img2,
                       window,
                       padding=padding,
                       groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = (
        ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) /
        ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    )

    mask = (mu1 != 0) * (mu2 != 0) if ignore_zeros else True

    if size_average:
        return ssim_map[mask].mean()
    else:
        return ssim_map[mask].mean(1).mean(1).mean(1)


class SSIM3D(torch.nn.Module):
    def __init__(self, window_size=(3, 11, 11), size_average=True):
        super(SSIM3D, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window3d(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _, _) = img1.size()

        if (channel == self.channel and
                self.window.data.type() == img1.data.type()):
            window = self.window
        else:
            window = create_window3d(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim3d(
            img1, img2, window, self.window_size, channel, self.size_average)


def ssim3d(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _, _) = img1.size()
    window = create_window3d(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim3d(img1, img2, window, window_size, channel, size_average)
