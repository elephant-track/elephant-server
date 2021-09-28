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
"""Implementations of PyTorch loss modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from elephant.util.pytorch_ssim import SSIM
from elephant.util.pytorch_ssim import SSIM3D
from elephant.util.gaussian_smoothing import GaussianSmoothing


class SegmentationLoss(nn.Module):
    """Loss for a segmentation model.

    Args:
        class_weights (Tensor, optional): class weightsss for `nn.NLLLoss`.
        false_weight (float): a weight for voxels with `False` labels.
            It is relative to a weight for voxels with `True` labels.
            The `True`/`False` weights are averaged with normalization across
            the voxels if `is_oldloss` = `False`, otherwise they are averaged
            without normalization.
        eps (float): term added to the denominator to improve numerical
            stability used in the Center Dice loss calculation.
        center_value: a value of the `center` label in the label tensor, used
            in the Center Dice loss calculation.
        is_oldloss: `True` if it is a loss without `True`/`False` weight
            normalization (shown in the paper), `False` otherwise (recommended).
        is_3d: `True` if 3D.
    """

    def __init__(self, class_weights=None, false_weight=10., eps=1e-6,
                 center_value=2, is_oldloss=False, is_3d=True):
        super().__init__()
        self.nll = nn.NLLLoss(weight=class_weights, reduction='mean')
        self.false_weight = false_weight
        self.eps = eps
        self.center_value = center_value
        self.is_oldloss = is_oldloss
        kernel_size = (3, 5, 5)
        sigma = (1., 8., 8.)
        padding = (1, 2, 2)
        n_dims = 2 + is_3d  # 3 or 2
        self.downsample = GaussianSmoothing(channels=1,
                                            kernel_size=kernel_size[-n_dims:],
                                            sigma=sigma[-n_dims:],
                                            dim=n_dims,
                                            stride=2,
                                            padding=padding[-n_dims:])
        self.upsmooth = GaussianSmoothing(channels=1,
                                          kernel_size=kernel_size[-n_dims:],
                                          sigma=sigma[-n_dims:],
                                          dim=n_dims,
                                          stride=1,
                                          padding=padding[-n_dims:])
        self.n_dims = n_dims

    def forward(self, prediction, target):
        assert prediction.shape[0] == target.shape[0]
        assert len(target.shape) == len(prediction.shape) - 1, (
            'target.shape: {}, prediction.shape: {}.'
            .format(target.shape, prediction.shape))
        assert target.max() != -1
        if hasattr(self, 'center_loss'):
            last_center_loss = self.center_loss.clone().detach()
        else:
            last_center_loss = torch.tensor(0.0)
        last_center_loss = last_center_loss.to(prediction.device)
        self.nll_loss = 0.0
        self.center_loss = 0.0
        self.smooth_loss = 0.0
        count = 0
        for i in range(target.shape[0]):
            if target[i].max() != -1:
                # NLL loss calculation
                light_weight_mask = (-1 < target[i]) * (target[i] < 3)
                nll_sum = 0
                weight_sum = 0
                light_weight_voxels = light_weight_mask.sum()
                if 0 < light_weight_voxels:
                    nll_sum += self.nll(
                        prediction[i:i + 1, :, light_weight_mask],
                        target[i:i + 1, light_weight_mask]
                    ) * light_weight_voxels
                    weight_sum += light_weight_voxels
                heavy_weight_mask = 2 < target[i]
                heavy_weight_voxels = heavy_weight_mask.sum()
                if 0 < heavy_weight_voxels:
                    nll_sum += self.nll(
                        prediction[i:i + 1, :, heavy_weight_mask],
                        torch.fmod(target[i:i + 1, heavy_weight_mask], 3)
                    ) * self.false_weight * heavy_weight_voxels
                    weight_sum += heavy_weight_voxels * self.false_weight
                if not self.is_oldloss and 0 < weight_sum:
                    self.nll_loss += nll_sum / weight_sum
                # Center Dice loss calculation
                if 2 in target[i] or 5 in target[i]:
                    center_loss_sum = 0
                    if 0 < light_weight_voxels:
                        center_pred = torch.exp(
                            prediction[i, -1, light_weight_mask])
                        center_target = (
                            target[i, light_weight_mask] == self.center_value
                        ).float()
                        intersection = (center_pred * center_target).sum()
                        denominator = (
                            (center_pred * center_pred).sum() +
                            (center_target * center_target).sum()
                        ).clamp(min=self.eps)
                        center_loss_sum += (
                            1 - (2 * intersection / denominator)
                        ) * light_weight_voxels
                    if 0 < heavy_weight_voxels:
                        center_pred = torch.exp(
                            prediction[i, -1, heavy_weight_mask])
                        center_target = (
                            torch.fmod(
                                target[i, heavy_weight_mask], 3
                            ) == self.center_value
                        ).float()
                        intersection = (center_pred * center_target).sum()
                        denominator = (
                            (center_pred * center_pred).sum() +
                            (center_target * center_target).sum()
                        ).clamp(min=self.eps)
                        center_loss_sum += (
                            1 - (2 * intersection / denominator)
                        ) * heavy_weight_voxels * self.false_weight
                    if not self.is_oldloss and 0 < weight_sum:
                        self.center_loss += center_loss_sum / weight_sum
                else:
                    self.center_loss += last_center_loss
                # Smooth loss calculation
                self.smooth_loss += abs(
                    torch.exp(prediction[i:i+1, -1:]) -
                    self.upsmooth(
                        F.interpolate(
                            self.downsample(
                                torch.exp(prediction[i:i+1, -1:])
                            ),
                            size=prediction.shape[-self.n_dims:],
                            mode='nearest',
                        )
                    )
                ).mean()
                count += 1
        assert count != 0
        self.nll_loss /= count
        self.center_loss /= count
        self.smooth_loss /= count
        return (self.nll_loss * 1 / 7 +
                self.center_loss * 5 / 7 +
                self.smooth_loss * 1 / 7)


class AutoencoderLoss(nn.Module):
    """Loss for a prior training for a segmentation model.

    Args:
        criterion (nn.Module, optional): a loss module for calculating the
            differences between prediction and target. The criterion should
            output tensors with the same size as inputs.
    """

    def __init__(self, criterion=nn.L1Loss()):
        super().__init__()
        self.criterion = criterion

    def forward(self, prediction, target):
        assert prediction.shape[0] == target.shape[0]
        assert len(target.shape) == len(prediction.shape), (
            'target.shape: {}, prediction.shape: {}.'
            .format(target.shape, prediction.shape))
        pred_exp = torch.exp(prediction)
        loss = (self.criterion(pred_exp[:, 2], target[:, 0]) +
                self.criterion(1 - pred_exp[:, 0], target[:, 0]))
        return loss


class FlowLoss(nn.Module):
    """Loss for a flow model.

    Args:
        criterion (nn.Module, optional): a loss module for calculating the
            differences between prediction and target. The criterion should
            output tensors with the same size as inputs.
        flow_norm_factor (Tuple[float, float, float], optional): values to be
            used to normalize (in the rangge [-1, 1]) the flow in each
            dimension. The values represent the maximum displacements the model
            can predict. Ther order is (X, Y, Z).
        dim_weights (Tuple[float, float, float], optional): weights for each
            dimension. For example, if the displacements in a specific
            dimension is smaller, the weight can be set larger than other
            dimensions. The weights are normalized such that they sum up to 1.
            The order is (Z, Y, X).
        is_oldloss: `True` if it is a loss without weight normalization
            (shown in the paper), `False` otherwise (recommended).
        is_3d: `True` if 3D.
    Channel order of target:
        (flow_x, flow_y, flow_z, mask, input_t0, input_t1)
    """

    def __init__(self,
                 criterion=nn.L1Loss(reduction='none'),
                 flow_norm_factor=None,
                 dim_weights=None,
                 is_oldloss=False,
                 is_3d=True):
        super().__init__()
        self.criterion = criterion
        n_dims = 2 + is_3d  # 3 or 2
        self.n_dims = n_dims
        if is_3d:
            self.ssim = SSIM3D(window_size=(3, 7, 7))
            if flow_norm_factor is None:
                flow_norm_factor = (80, 80, 10)
            if dim_weights is None:
                dim_weights = (1./3, 1./3, 1./3)
        else:
            self.ssim = SSIM(window_size=7)
            if flow_norm_factor is None:
                flow_norm_factor = (80, 80)
            if dim_weights is None:
                dim_weights = (1./2, 1./2)
        self.is_oldloss = is_oldloss
        self.flow_norm_factor = flow_norm_factor  # X, Y(, Z)
        if not is_oldloss:
            dim_weights = tuple(w / sum(dim_weights) for w in dim_weights)
        self.dim_weights = dim_weights
        kernel_size = (3, 5, 5)
        sigma = (1., 8., 8.)
        padding = (1, 2, 2)
        self.downsample = GaussianSmoothing(channels=n_dims,
                                            kernel_size=kernel_size[-n_dims:],
                                            sigma=sigma[-n_dims:],
                                            dim=n_dims,
                                            stride=2,
                                            padding=padding[-n_dims:])
        self.upsmooth = GaussianSmoothing(channels=n_dims,
                                          kernel_size=kernel_size[-n_dims:],
                                          sigma=sigma[-n_dims:],
                                          dim=n_dims,
                                          stride=1,
                                          padding=padding[-n_dims:])

    def forward(self, prediction, target):
        assert prediction.shape == target[:, :self.n_dims].shape, (
            'prediction.shape {} should be same as target[:,:{:d}].shape {}'
            .format(prediction.shape, self.n_dims,
                    target[:, :self.n_dims].shape))
        if self.is_oldloss:
            cr_loss = self.criterion(
                prediction, target[:, :self.n_dims]) * target[:, -3:-2]
            self.instance_loss = sum(
                cr_loss[:, d].mean() * self.dim_weights[d]
                for d in range(self.n_dims)
            )
        else:
            cr_loss = self.criterion(prediction, target[:, :self.n_dims])
            self.instance_loss = sum(
                cr_loss[:, d][0 < target[:, -3]].mean() * self.dim_weights[d]
                for d in range(self.n_dims)
            )
        dims_order = [0, 2, 3, 1]
        if self.n_dims == 3:
            dims_order.insert(3, 4)
        flow = prediction.clone().permute(*dims_order).to(prediction.device)
        flow[..., 0] *= self.flow_norm_factor[0] / (flow.shape[-2] / 2)
        flow[..., 1] *= self.flow_norm_factor[1] / (flow.shape[-3] / 2)
        if self.n_dims == 3:
            flow[..., 2] *= self.flow_norm_factor[2] / (flow.shape[-4] / 2)
        grid = torch.zeros(flow.shape,
                           dtype=torch.float32
                           ).to(prediction.device)
        grid[..., 0] = torch.linspace(-1, 1,
                                      flow.shape[-2]
                                      )
        grid[..., 1] = torch.linspace(-1, 1,
                                      flow.shape[-3]
                                      ).view(flow.shape[-3], 1)
        if self.n_dims == 3:
            grid[..., 2] = torch.linspace(-1, 1,
                                          flow.shape[-4]
                                          ).view(flow.shape[-4], 1, 1)
        warp = torch.nn.functional.grid_sample(target[:, -1:],
                                               grid - flow,
                                               align_corners=True)
        self.ssim_loss = 1 - self.ssim(target[:, -2:-1], warp)
        self.smooth_loss = abs(
            prediction -
            self.upsmooth(
                F.interpolate(
                    self.downsample(prediction),
                    size=prediction.shape[-self.n_dims:],
                    mode='nearest',
                )
            )
        ).mean()

        if self.is_oldloss:
            return (
                self.instance_loss +
                self.ssim_loss * 1e-4 +
                self.smooth_loss * 1e-4
            )

        loss_weights = [1.0, 1e-2, 1e-2]
        loss_weights = [w / sum(loss_weights) for w in loss_weights]

        return (
            self.instance_loss * loss_weights[0] +
            self.ssim_loss * loss_weights[1] +
            self.smooth_loss * loss_weights[2]
        )
