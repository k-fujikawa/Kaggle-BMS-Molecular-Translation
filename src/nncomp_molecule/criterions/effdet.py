import effdet
import numpy as np
import torch
import torch.nn.functional as F
from effdet.anchors import Anchors, AnchorLabeler, generate_detections, MAX_DETECTION_POINTS
from effdet.loss import focal_loss_legacy, _box_loss
from typing import Optional, List, Tuple

import nncomp.registry as R


MAX_CLASSES = 1000


def loss_fn(
    cls_outputs: List[torch.Tensor],
    box_outputs: List[torch.Tensor],
    cls_targets: List[torch.Tensor],
    box_targets: List[torch.Tensor],
    num_positives: torch.Tensor,
    num_classes: int,
    alpha: float,
    gamma: float,
    delta: float,
    box_loss_weight: float,
    lookup_table: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes total detection loss.
    Computes total detection loss including box and class loss from all levels.
    Args:
        cls_outputs: a List with values representing logits in [batch_size, height, width, num_anchors].
            at each feature level (index)
        box_outputs: a List with values representing box regression targets in
            [batch_size, height, width, num_anchors * 4] at each feature level (index)
        cls_targets: groundtruth class targets.
        box_targets: groundtrusth box targets.
        num_positives: num positive grountruth anchors
    Returns:
        total_loss: an integer tensor representing total loss reducing from class and box losses from all levels.
        cls_loss: an integer tensor representing total class loss.
        box_loss: an integer tensor representing total box regression loss.
    """
    # Sum all positives in a batch for normalization and avoid zero
    # num_positives_sum, which would lead to inf loss during training
    num_positives_sum = (num_positives.sum() + 1.0).float()
    levels = len(cls_outputs)

    cls_losses = []
    box_losses = []
    for l in range(levels):
        cls_targets_at_level = cls_targets[l]
        box_targets_at_level = box_targets[l]

        # Onehot encoding for classification labels.
        cls_targets_at_level_oh = lookup_table[cls_targets_at_level]
        bs, height, width, _, _ = cls_targets_at_level_oh.shape
        cls_targets_at_level_oh = cls_targets_at_level_oh.view(bs, height, width, -1)
        cls_outputs_at_level = cls_outputs[l].permute(0, 2, 3, 1).float()
        cls_loss = focal_loss_legacy(
            cls_outputs_at_level, cls_targets_at_level_oh,
            alpha=alpha, gamma=gamma, normalizer=num_positives_sum)
        cls_loss = cls_loss.view(bs, height, width, -1, num_classes)
        cls_loss = cls_loss * (cls_targets_at_level != -2).unsqueeze(-1)
        cls_losses.append(cls_loss.sum())   # FIXME reference code added a clamp here at some point ...clamp(0, 2))
        box_losses.append(_box_loss(
            box_outputs[l].permute(0, 2, 3, 1).float(),
            box_targets_at_level,
            num_positives_sum,
            delta=delta))

    # Sum per level losses to total loss.
    cls_loss = torch.sum(torch.stack(cls_losses, dim=-1), dim=-1)
    box_loss = torch.sum(torch.stack(box_losses, dim=-1), dim=-1)
    total_loss = cls_loss + box_loss_weight * box_loss
    return total_loss, cls_loss, box_loss


loss_jit = torch.jit.script(loss_fn)


@R.CriterionRegistry.add
class EffdetLoss(torch.nn.Module):
    def __init__(
        self,
        model_name,
        image_size,
        classification_tasks,
    ):
        super().__init__()
        config = effdet.get_efficientdet_config(model_name)
        config.image_size = image_size
        self.config = config
        self.anchors = Anchors(
            config.min_level, config.max_level,
            config.num_scales, config.aspect_ratios,
            config.anchor_scale, config.image_size)
        self.alpha = config.alpha
        self.gamma = config.gamma
        self.delta = config.delta
        self.box_loss_weight = config.box_loss_weight
        self.classification_tasks = classification_tasks
        self.anchor_labelers = {
            task: AnchorLabeler(
                self.anchors,
                num_classes,
                match_threshold=0.5
            )
            for task, num_classes in classification_tasks.items()
        }
        self.lookup_tables = {
            task: np.zeros((num_classes ** 2, num_classes)).astype("f")
            for task, num_classes in classification_tasks.items()
        }
        for task, num_classes in classification_tasks.items():
            for i in range(len(self.lookup_tables[task])):
                self.lookup_tables[task][i, i % num_classes] = 1
                if i >= num_classes:
                    self.lookup_tables[task][i, i // num_classes] = 1
            self.lookup_tables[task][-1] = 0

    def forward(self, y_bboxes, bboxes, **kwargs):
        device = y_bboxes[0].device
        self.to(device)
        cls_outputs = {k: kwargs[k] for k in sorted(kwargs) if k.startswith("y_")}
        cls_targets = {k: kwargs[k] for k in sorted(kwargs) if not k.startswith("y_")}
        assert all([y.replace("y_", "") == t for y, t in zip(cls_outputs, cls_targets)])
        cls_losses = {}
        box_losses = {}

        for (task, _cls_targets), _cls_outputs in \
                zip(cls_targets.items(), cls_outputs.values()):
            num_classes = self.classification_tasks[task]
            _cls_targets, box_targets, num_positives = self.anchor_labelers[task].batch_label_anchors(
                bboxes, _cls_targets
            )
            lookup_table = torch.tensor(self.lookup_tables[task]).to(device)
            total_loss, cls_loss, box_loss = loss_jit(
                cls_outputs=_cls_outputs,
                box_outputs=y_bboxes,
                cls_targets=_cls_targets,
                box_targets=box_targets,
                num_positives=num_positives,
                num_classes=num_classes,
                alpha=self.alpha,
                gamma=self.gamma,
                delta=self.delta,
                box_loss_weight=self.box_loss_weight,
                lookup_table=lookup_table,
            )
            cls_losses[task] = cls_loss
            box_losses[task] = box_loss

        loss = self.box_loss_weight * box_loss \
            + torch.stack(list(cls_losses.values())).mean()

        return dict(
            loss=loss,
            box_loss=box_loss,
            **cls_losses,
        )
