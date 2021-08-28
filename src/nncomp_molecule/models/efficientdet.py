import effdet
import torch
import timm
from effdet.efficientdet import (
    get_feature_info,
    _init_weight,
    _init_weight_alt,
    BiFpn,
    HeadNet,
)

import nncomp.registry as R


@R.ModuleRegistry.add
@R.ModelRegistry.add
class EfficientDetModel(torch.nn.Module):
    def __init__(
        self,
        model_name,
        image_size,
        classification_tasks,
        pretrained=True,
        norm_kwargs=None,
        alternate_init=False,
    ):
        super().__init__()
        norm_kwargs = norm_kwargs or dict(eps=.001, momentum=.01)
        config = effdet.get_efficientdet_config(model_name)
        config.image_size = image_size
        self.config = config
        self.config.norm_kwargs = norm_kwargs
        self.backbone = timm.create_model(
            config.backbone_name,
            features_only=True,
            out_indices=(2, 3, 4),
            pretrained=pretrained,
            **config.backbone_args,
        )
        feature_info = get_feature_info(self.backbone)
        self.fpn = BiFpn(
            config,
            feature_info,
        )
        self.box_net = HeadNet(
            config,
            num_outputs=4,
        )
        self.class_nets = torch.nn.ModuleDict({
            task: HeadNet(
                config,
                num_outputs=num_outputs,
            )
            for task, num_outputs in classification_tasks.items()
        })
        for n, m in self.named_modules():
            if 'backbone' not in n:
                if alternate_init:
                    _init_weight_alt(m, n)
                else:
                    _init_weight(m, n)

    def forward(self, **batch):
        x = batch["image"]
        x = self.backbone(x)
        x = self.fpn(x)
        y_bboxes = self.box_net(x)
        y_labels = {
            f"y_{task}": net(x)
            for task, net in self.class_nets.items()
        }
        return dict(
            y_bboxes=y_bboxes,
            **y_labels,
        )
