from typing import List

import numpy as np
import torch
import timm
import nncomp.registry as R


@R.ModuleRegistry.add
class TimmModelWrapper(torch.nn.Module):
    def __init__(
        self,
        model_name: str,
        pretrained=False,
        num_classes=1000,
        in_chans=3,
        checkpoint_path="",
        position_embedding_size: List[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_chans,
            checkpoint_path=checkpoint_path,
            **kwargs,
        )
        self.position_embedding_size = position_embedding_size
        if position_embedding_size is not None:
            self.position_embedding = torch.nn.Parameter(torch.Tensor(
                *position_embedding_size,
            ))
            torch.nn.init.normal_(self.position_embedding)

    def forward_features(self, x):
        h = self.model.forward_features(x)
        if self.position_embedding_size is not None:
            h_position = self.position_embedding.expand(
                len(h),
                *self.position_embedding.shape,
            )
            h += h_position
        return h


@R.ModuleRegistry.add
class VisionTransformerWrapper(torch.nn.Module):
    def __init__(
        self,
        model_name: str,
        pretrained=False,
        num_classes=1000,
        in_chans=3,
        checkpoint_path="",
        **kwargs,
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_chans,
            checkpoint_path=checkpoint_path,
            **kwargs,
        )

    def forward_features(self, x):
        x = self.model.patch_embed(x)
        cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.model.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.model.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.model.pos_drop(x + self.model.pos_embed)
        x = self.model.blocks(x)
        x = self.model.norm(x)
        return x
