# Copyright (c) 2019 Shigeki Karita
#               2020 Mobvoi Inc (Binbin Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Label smoothing module."""

import math
import torch
from torch import nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):
    """Label-smoothing loss.

    In a standard CE loss, the label's data distribution is:
    [0,1,2] ->
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]

    In the smoothing version CE Loss,some probabilities
    are taken from the true label prob (1.0) and are divided
    among other labels.

    e.g.
    smoothing=0.1
    [0,1,2] ->
    [
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9],
    ]

    Args:
        size (int): the number of class
        padding_idx (int): padding class id which will be ignored for loss
        smoothing (float): smoothing rate (0.0 means the conventional CE)
        normalize_length (bool):
            normalize loss by sequence length if True
            normalize loss by batch size if False
    """

    def __init__(self,
                 size: int,
                 padding_idx: int,
                 smoothing: float,
                 normalize_length: bool = False):
        """Construct an LabelSmoothingLoss object."""
        super(LabelSmoothingLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="none")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.normalize_length = normalize_length
        self.kl_constant = 0.0
        if self.confidence > 0.0:
            self.kl_constant += self.confidence * math.log(self.confidence)
        if self.smoothing > 0.0:
            self.kl_constant += self.smoothing * math.log(self.smoothing / (self.size - 1))

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss between x and target.

        The model outputs and data labels tensors are flatten to
        (batch*seqlen, class) shape and a mask is applied to the
        padding part which should not be calculated for loss.

        Args:
            x (torch.Tensor): prediction (batch, seqlen, class)
            target (torch.Tensor):
                target signal masked with self.padding_id (batch, seqlen)
        Returns:
            loss (torch.Tensor) : The KL loss, scalar float value
        """
        assert x.size(2) == self.size
        batch_size = x.size(0)
        x = x.view(-1, self.size)
        target = target.view(-1)
        ignore = target == self.padding_idx  # (N,)
        total = len(target) - ignore.sum().item()

        # Fast path for plain cross entropy (most common for CosyVoice recipes).
        # This reduces peak memory on small VRAM GPUs and avoids extra tensor
        # allocations in the smoothing implementation.
        if self.smoothing == 0.0:
            loss_sum = F.cross_entropy(
                x,
                target,
                ignore_index=self.padding_idx,
                reduction="sum",
            )
            denom = total if self.normalize_length else batch_size
            # Avoid division by zero in pathological cases (all padding).
            if denom == 0:
                return loss_sum * 0.0
            return loss_sum / denom

        # For smoothing > 0, avoid -1 index for gather.
        target = target.masked_fill(ignore, 0)

        # Memory-efficient implementation: avoid allocating a dense (N, V)
        # target distribution (true_dist). This matters a lot for large vocab.
        #
        # For smoothing == 0, this reduces to NLL loss:
        #   loss = -log_softmax(x)[target]
        logsumexp = torch.logsumexp(x, dim=1).float()  # (N,)
        target_logits = x.gather(1, target.unsqueeze(1)).squeeze(1).float()  # (N,)
        log_probs_y = target_logits - logsumexp  # (N,)
        nll_loss = -log_probs_y

        if self.smoothing == 0.0:
            loss = nll_loss
        else:
            # Smooth loss for uniform distribution over non-target classes:
            #   smooth_loss = -mean_{j!=y} log_softmax(x)[j]
            sum_log_probs = x.sum(dim=1, dtype=torch.float32) - self.size * logsumexp  # (N,)
            smooth_loss = -(sum_log_probs - log_probs_y) / (self.size - 1)
            loss = self.confidence * nll_loss + self.smoothing * smooth_loss

            # Match the original KLDivLoss(true_dist || log_softmax(x)) up to
            # a constant that does not affect gradients.
            if self.kl_constant != 0.0:
                loss = loss + loss.new_tensor(self.kl_constant)

        loss = loss.masked_fill(ignore, 0.0)
        denom = total if self.normalize_length else batch_size
        return loss.sum() / denom
