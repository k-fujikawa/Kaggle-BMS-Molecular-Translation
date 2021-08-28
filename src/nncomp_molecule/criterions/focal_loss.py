import torch

import nncomp.registry as R


@R.CriterionRegistry.add
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.lossfunc = torch.nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        n_output = input.shape[-1]
        one_hot_target = torch.eye(n_output, device=input.device)[target]
        bceloss = self.lossfunc(input, one_hot_target)

        probs = torch.sigmoid(input)
        probs_gt = torch.where(
            one_hot_target == 1,
            probs,
            1 - probs,
        )
        modulator = torch.pow(1 - probs_gt, self.gamma)
        weighted_loss = torch.where(
            one_hot_target == 1,
            self.alpha * modulator * bceloss,
            (1 - self.alpha) * modulator * bceloss
        )
        weighted_loss = torch.where(
            (target != self.ignore_index)[:, :, None].expand(input.shape),
            weighted_loss,
            torch.zeros_like(weighted_loss),
        )
        weighted_loss = weighted_loss.sum(dim=(1, 2))
        weighted_loss = weighted_loss.mean()

        return weighted_loss


@R.CriterionRegistry.add
class FocalLossEx(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.lossfunc = torch.nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        n_output = input.shape[-1]
        one_hot_target = torch.eye(n_output, device=input.device)[target]
        bceloss = self.lossfunc(input, one_hot_target)

        probs = torch.sigmoid(input)
        probs_gt = torch.where(
            one_hot_target == 1,
            probs,
            1 - probs,
        )
        modulator = torch.pow(1 - probs_gt, self.gamma)
        breakpoint()
        weighted_loss = torch.where(
            one_hot_target == 1,
            self.alpha * modulator * bceloss,
            (1 - self.alpha) * modulator * bceloss
        )
        weighted_loss = torch.where(
            (target != self.ignore_index)[:, :, None].expand(input.shape),
            weighted_loss,
            torch.zeros_like(weighted_loss),
        )
        weighted_loss = weighted_loss.sum(dim=(1, 2))
        weighted_loss = weighted_loss.mean()

        return weighted_loss
