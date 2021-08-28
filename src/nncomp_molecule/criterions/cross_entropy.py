import torch
import torch.nn as nn
import torch.nn.functional as F

import nncomp.registry as R


@R.CriterionRegistry.add
class FixedCrossEntropy2D(nn.Module):
    def __init__(self, eps=0.1, ignore_index=0):
        super().__init__()
        self.eps = eps
        self.ignore_index = ignore_index
        self.lossfunc = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            reduction="none",
        )

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        batch_size, seqlen, n_output = input.shape
        input = input.reshape(-1, n_output)
        target = target.reshape(-1)
        mask = target != self.ignore_index
        device = input.device

        y_prob = F.softmax(input, dim=-1)
        gt_prob = y_prob[range(len(target)), target]
        mask *= gt_prob < (1 - self.eps)
        loss = self.lossfunc(input, target)
        loss = loss.where(mask, torch.tensor(0., device=device))
        loss = loss.reshape(batch_size, seqlen)
        loss = loss.sum(dim=1)
        n_loss_targets = mask.reshape(batch_size, seqlen).sum(dim=1)
        n_loss_targets = n_loss_targets.where(
            n_loss_targets > 0,
            torch.ones_like(n_loss_targets),
        )
        loss = (loss / n_loss_targets).mean()
        return loss
