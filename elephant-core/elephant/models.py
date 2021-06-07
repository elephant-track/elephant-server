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
"""Implementations of PyTorch models."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from elephant.util import get_next_multiple


class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest',
                 align_corners=None):
        super(Interpolate, self).__init__()
        self.interpolate = F.interpolate
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interpolate(x,
                             size=self.size,
                             scale_factor=self.scale_factor,
                             mode=self.mode,
                             align_corners=self.align_corners)
        return x


class UNet(nn.Module):
    """ UNet 2D/3D implementation
    Arguments:
      in_channels: number of input channels
      out_channels: number of output channels
      final_activation: activation applied to the network output
      keep_axials: specify levels to keep axials (only used for 3D)
      is_pad: True if padding is used
      is_3d: True if 3D
    """

    # _conv_block, _pooler and _upsampler are just helper functions to
    # construct the model.
    # encapsulating them like so also makes it easy to re-use
    # the model implementation with different architecture elements

    # Convolutional block for single layer of the decoder / encoder
    # we apply to 2d/3d convolutions with relu activation
    def _conv_block(self, in_channels, out_channels):
        n_groups = min(32, out_channels)
        return nn.Sequential(
            self.conv(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.GroupNorm(n_groups, out_channels),
            self.conv(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.GroupNorm(n_groups, out_channels),
        )

    def _encoder_block(self, in_channels, out_channels):
        return self._conv_block(in_channels, out_channels)

    def _decoder_block(self, in_channels, out_channels):
        return self._conv_block(in_channels, out_channels)

    # downsampling via MaxPooling 2d/3d
    def _pooler(self, keep_axial=False):
        if self.n_dims == 2:
            return nn.MaxPool2d(kernel_size=(2, 2))
        elif keep_axial:
            return nn.MaxPool3d(kernel_size=(1, 2, 2))
        return nn.MaxPool3d(kernel_size=(2, 2, 2))

    # upsampling via interpolate function
    def _upsampler(self, keep_axial=False):
        if self.n_dims == 2:
            return Interpolate(scale_factor=(2, 2))
        elif keep_axial:
            return Interpolate(scale_factor=(1, 2, 2))
        return Interpolate(scale_factor=(2, 2, 2))

    def __init__(self, in_channels=1, out_channels=1, final_activation=None,
                 keep_axials=(True, True, True, False), is_pad=False,
                 is_3d=True):
        super().__init__()

        # the depth (= number of encoder / decoder levels) is
        # hard-coded to 4
        self.depth = 4

        self.is_3d = is_3d
        self.n_dims = 2 + self.is_3d
        if is_3d:
            self.conv = nn.Conv3d
            self.base_size = (2**keep_axials.count(False), 16, 16)
            self.interpolate_mode = 'trilinear'
        else:
            self.conv = nn.Conv2d
            self.base_size = (16, 16)
            self.interpolate_mode = 'bilinear'

        # pad edges if specified
        self.is_pad = is_pad
        # the order for pad size is (left, right, top, bottom, front, back)
        self.pad_size = sum([[x//2, ] * 2 for x in self.base_size[::-1]], [])

        # the final activation must either be None or a Module
        if final_activation is not None:
            assert isinstance(final_activation,
                              nn.Module), "Activation must be torch module"

        # all lists of conv layers (or other nn.Modules with parameters) must
        # be wraped into a nn.ModuleList

        # modules of the encoder path
        self.encoder = nn.ModuleList([
            self._encoder_block(in_channels, 16),
            self._encoder_block(16, 32),
            self._encoder_block(32, 64),
            self._encoder_block(64, 128)
        ])
        # the base convolution block
        self.base = self._encoder_block(128, 256)
        # modules of the decoder path
        self.decoder = nn.ModuleList([
            self._decoder_block(384, 128),
            self._decoder_block(192, 64),
            self._decoder_block(96, 32),
            self._decoder_block(48, 16)
        ])
        # the pooling layers; we use 2x2 MaxPooling
        self.poolers = nn.ModuleList([
            self._pooler(keep_axial=keep_axials[0]),
            self._pooler(keep_axial=keep_axials[1]),
            self._pooler(keep_axial=keep_axials[2]),
            self._pooler(keep_axial=keep_axials[3])
        ])
        # the upsampling layers
        self.upsamplers = nn.ModuleList([
            self._upsampler(keep_axial=keep_axials[3]),
            self._upsampler(keep_axial=keep_axials[2]),
            self._upsampler(keep_axial=keep_axials[1]),
            self._upsampler(keep_axial=keep_axials[0])
        ])
        # output conv and activation
        # the output conv is not followed by a non-linearity, because we apply
        # activation afterwards
        self.out_conv = self.conv(16, out_channels, 1)
        self.activation = final_activation

        # initialize parameters with kaiming_normal
        for m in self.modules():
            if isinstance(m, self.conv):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    def _conform(self, input):
        self.org_size = input.size()[-self.n_dims:]
        dst_size = [get_next_multiple(s, base) for s, base in
                    zip(self.org_size, self.base_size)]
        if list(self.org_size) == dst_size:
            return input, False
        return (F.interpolate(input,
                              dst_size,
                              mode=self.interpolate_mode,
                              align_corners=True),
                True)

    def forward(self, input):
        # pad if specified
        if self.is_pad:
            input = F.pad(input, self.pad_size, 'replicate')
        # resize if required
        x, is_resized = self._conform(input)
        # apply encoder path
        encoder_out = []
        for level in range(self.depth):
            x = self.encoder[level](x)
            encoder_out.append(x)
            x = self.poolers[level](x)

        # apply base
        x = self.base(x)

        # apply decoder path
        encoder_out = encoder_out[::-1]
        for level in range(self.depth):
            x = self.upsamplers[level](x)
            x = self.decoder[level](torch.cat((encoder_out[level], x), dim=1))
        # apply output conv and activation (if given)
        x = self.out_conv(x)
        if self.activation is not None:
            x = self.activation(x)
        # put back to the original size
        if is_resized:
            x = F.interpolate(x,
                              self.org_size,
                              mode=self.interpolate_mode,
                              align_corners=True)
        if self.is_pad:
            slices = (slice(None), slice(None)) + tuple(
                slice(self.pad_size[-(2+2*i)], -self.pad_size[-(1+2*i)])
                for i in range(self.n_dims)
            )
            x = x[slices]
        return x

    @ classmethod
    def three_class_segmentation(cls, keep_axials=(False, False, False, False),
                                 is_eval=False, device=None, state_dict=None,
                                 is_decoder_only=False, is_pad=False,
                                 is_3d=True):
        model = cls(1, 3, final_activation=torch.nn.LogSoftmax(
            dim=1), keep_axials=keep_axials, is_pad=is_pad, is_3d=is_3d)
        if state_dict:
            model.load_state_dict(state_dict)
        if device:
            model.to(device)
        if is_eval:
            model.eval()
        else:
            model.train()
        if is_decoder_only:
            for param in model.encoder.parameters():
                param.requires_grad = False
        return model


class ResBlockFlow(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1,
                 activation=nn.LeakyReLU(0.1), is_3d=True):
        super().__init__()
        self.conv = (lambda args: nn.conv3d(*args) if is_3d else
                     lambda args: nn.conv2d(*args))
        self.res = self.conv(in_channels, out_channels,
                             kernel_size=1, bias=False, stride=stride)
        self.conv1 = self.conv(in_channels, out_channels,
                               kernel_size=3, padding=1, stride=stride)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = self.conv(out_channels, out_channels,
                               kernel_size=3, padding=1)
        self.activation = activation

    def forward(self, x):
        # TODO: check if division by 2 is required
        return self.activation(
            (self.res(x) + self.conv2(self.relu(self.conv1(x)))) / 2
        )


class FlowResNet(UNet):
    def _res_block(self, in_channels, out_channels,
                   activation=nn.LeakyReLU(0.1)):
        return ResBlockFlow(in_channels, out_channels, activation=activation,
                            is_3d=self.is_3d)

    def _conv_block(self, in_channels, out_channels,
                    activation=nn.LeakyReLU(0.1)):
        return nn.Sequential(
            self.conv(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            self.conv(out_channels, out_channels, kernel_size=3, padding=1),
            activation
        )

    def _encoder_block(self, in_channels, out_channels):
        return self._res_block(in_channels, out_channels)

    @ classmethod
    def three_dimensional_flow(cls, keep_axials=(False, False, False, False),
                               is_eval=False, device=None, state_dict=None,
                               is_decoder_only=False, is_3d=True):
        model = cls(2, 3, final_activation=nn.Tanh(), keep_axials=keep_axials,
                    is_pad=False, is_3d=is_3d)
        if state_dict:
            model.load_state_dict(state_dict)
        if device:
            model.to(device)
        if is_eval:
            model.eval()
        else:
            model.train()
        if is_decoder_only:
            for param in model.encoder.parameters():
                param.requires_grad = False
        return model


def load_seg_models(model_path, keep_axials, device, is_eval=False,
                    is_decoder_only=False, is_pad=False, is_3d=True):
    checkpoint = torch.load(model_path)
    state_dicts = checkpoint if isinstance(checkpoint, list) else [checkpoint]
    # print(len(state_dicts), 'models will be ensembled')
    models = [UNet.three_class_segmentation(
        keep_axials,
        is_eval=is_eval,
        device=device,
        state_dict=state_dict,
        is_decoder_only=is_decoder_only,
        is_pad=is_pad,
        is_3d=is_3d,
    ) for state_dict in state_dicts]
    return models


def load_flow_models(model_path, keep_axials, device, is_eval=False,
                     is_decoder_only=False, is_pad=False, is_3d=True):
    checkpoint = torch.load(model_path)
    state_dicts = checkpoint if isinstance(checkpoint, list) else [checkpoint]
    # print(len(state_dicts), 'models will be ensembled')
    return [FlowResNet.three_dimensional_flow(
        keep_axials=keep_axials,
        is_eval=is_eval,
        device=device,
        state_dict=state_dict,
        is_decoder_only=is_decoder_only,
        is_pad=is_pad,
        is_3d=is_3d,
    ) for state_dict in state_dicts]
