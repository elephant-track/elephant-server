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

"""Tests for the segmentation loss."""
import torch

from elephant.losses import SegmentationLoss


def test_loss_with_corrections_next_to_ordinary_labels():
    # Targets are labels - 1: -1 unlabelled, 0..2 background/border/centre,
    # 3..5 the same classes as corrections (fp/fn), weighted by false_weight.
    # A crop holding both used to raise "result type Float can't be cast to
    # the desired output type Long", so no correction was ever trained on.
    torch.manual_seed(0)
    target = torch.full((1, 4, 16, 16), -1, dtype=torch.long)
    target[0, :, 2:6, 2:6] = 1
    target[0, :, 3:5, 3:5] = 2
    target[0, :, 10:14, 10:14] = 3  # fp: background as a correction
    prediction = torch.log_softmax(
        torch.randn(1, 3, 4, 16, 16, requires_grad=True), dim=1)
    loss_fn = SegmentationLoss(class_weights=torch.tensor([1., 10., 10.]),
                               false_weight=10., is_3d=True)
    loss = loss_fn(prediction, target)
    assert torch.isfinite(loss)
    loss.backward()
