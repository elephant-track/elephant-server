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
                 is_pad=False, is_3d=True):
        super().__init__()

        # the depth (= number of encoder / decoder levels) is
        # hard-coded to 4
        self.depth = 4

        self.is_3d = is_3d
        self.n_dims = 2 + self.is_3d
        if is_3d:
            self.conv = nn.Conv3d
            self.interpolate_mode = 'trilinear'
        else:
            self.conv = nn.Conv2d
            self.interpolate_mode = 'bilinear'

        # pad edges if specified
        self.is_pad = is_pad

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

    def _conform(self, input, base_size):
        self.org_size = input.size()[-self.n_dims:]
        dst_size = [get_next_multiple(s, base) for s, base in
                    zip(self.org_size, base_size)]
        if list(self.org_size) == dst_size:
            return input, False
        return (F.interpolate(input,
                              dst_size,
                              mode=self.interpolate_mode,
                              align_corners=True),
                True)

    def forward(self, input, keep_axials):
        n_batches = input.shape[0]
        if self.is_3d:
            z_poolings = max([int(len(ka) - ka.sum()) for ka in keep_axials])
            base_size = (2**z_poolings, 16, 16)
        else:
            base_size = (16, 16)
        # resize if required
        x, is_resized = self._conform(input, base_size)
        # pad if specified
        if self.is_pad:
            # the order for pad size is (left, right, top, bottom, front, back)
            pad_size = sum([[x//2, ] * 2 for x in base_size[::-1]], [])
            x = F.pad(x, pad_size, 'replicate')
        # apply encoder path
        encoder_out = []
        for level in range(self.depth):
            x = self.encoder[level](x)
            encoder_out.append(x)
            # the pooling layers; we use 2x2 MaxPooling
            x = torch.cat([
                self._pooler(keep_axials[n, level])(x[n:n+1])
                for n in range(n_batches)
            ])

        # apply base
        x = self.base(x)

        # apply decoder path
        encoder_out = encoder_out[::-1]
        for level in range(self.depth):
            # the upsampling layers
            x = torch.cat([
                self._upsampler(keep_axials[n, -(1 + level)])(x[n:n+1])
                for n in range(n_batches)
            ])
            x = self.decoder[level](torch.cat((encoder_out[level], x), dim=1))

        # apply output conv and activation (if given)
        x = self.out_conv(x)
        if self.activation is not None:
            x = self.activation(x)
        # remove pad if specified
        if self.is_pad:
            slices = (slice(None), slice(None)) + tuple(
                slice(pad_size[-(2+2*i)], x.shape[2+i] - pad_size[-(1+2*i)])
                for i in range(self.n_dims)
            )
            x = x[slices]
        # put back to the original size
        if is_resized:
            x = F.interpolate(x,
                              self.org_size,
                              mode=self.interpolate_mode,
                              align_corners=True)
        return x

    @ classmethod
    def three_class_segmentation(cls, is_eval=False, device=None,
                                 state_dict=None, is_decoder_only=False,
                                 is_pad=False, is_3d=True):
        model = cls(1, 3, final_activation=torch.nn.LogSoftmax(
            dim=1), is_pad=is_pad, is_3d=is_3d)
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


class ResUNet(UNet):
    def _res_block(self, in_channels, out_channels,
                   activation=nn.LeakyReLU(0.1)):
        return ResBlock(in_channels, out_channels, activation=activation,
                        is_3d=self.is_3d)

    def _encoder_block(self, in_channels, out_channels):
        return self._res_block(in_channels, out_channels)

    @ classmethod
    def three_class_segmentation(cls, is_eval=False, device=None,
                                 state_dict=None, is_decoder_only=False,
                                 is_pad=False, is_3d=True):
        return super().three_class_segmentation(is_eval,
                                                device,
                                                state_dict,
                                                is_decoder_only,
                                                is_pad,
                                                is_3d)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1,
                 activation=nn.ReLU(), is_3d=True):
        super().__init__()
        n_groups = min(32, out_channels)
        self.conv = nn.Conv3d if is_3d else nn.Conv2d
        self.res = self.conv(in_channels, out_channels,
                             kernel_size=1, bias=False, stride=stride)
        self.conv1 = self.conv(in_channels, out_channels,
                               kernel_size=3, padding=1, stride=stride)
        self.relu = nn.ReLU()
        self.norm1 = nn.GroupNorm(n_groups, out_channels)
        self.conv2 = self.conv(out_channels, out_channels,
                               kernel_size=3, padding=1)
        self.activation = activation
        self.norm2 = nn.GroupNorm(n_groups, out_channels)

    def forward(self, x):
        return self.norm2(self.activation(
            (self.res(x) + self.conv2(self.norm1(self.relu(self.conv1(x)))))
        ))


class ResBlockFlow(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1,
                 activation=nn.LeakyReLU(0.1), is_3d=True):
        super().__init__()
        self.conv = nn.Conv3d if is_3d else nn.Conv2d
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
    def three_dimensional_flow(cls, is_eval=False, device=None, state_dict=None,
                               is_decoder_only=False, is_pad=False, is_3d=True):
        model = cls(2, 2 + is_3d, final_activation=nn.Tanh(),
                    is_pad=is_pad, is_3d=is_3d)
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
