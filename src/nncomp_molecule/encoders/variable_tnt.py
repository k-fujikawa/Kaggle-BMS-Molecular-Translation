import math
from numpy.core.numeric import flatnonzero

import torch
import torch.nn as nn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, trunc_normal_
from timm.models.vision_transformer import Mlp

import nncomp.registry as R
from nncomp.modules import PositionalEncoding2D


HPARAMS = {
    "tnt_s_patch16_224": {
        "patch_size": 16,
        "embed_dim": 384,
        "in_dim": 24,
        "depth": 12,
        "num_heads": 6,
        "in_num_head": 4,
        "qkv_bias": False,
        "pretrained_img_size": 224,
    }
}


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'pixel_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'tnt_s_patch16_224': _cfg(
        url='https://github.com/contrastive/pytorch-image-models/releases/download/TNT/tnt_s_patch16_224.pth.tar',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'tnt_b_patch16_224': _cfg(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}


class Attention(nn.Module):
    """ Multi-Head Attention"""
    def __init__(
        self,
        dim,
        hidden_dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.,
        proj_drop=0.
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        head_dim = hidden_dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.qk = nn.Linear(dim, hidden_dim * 2, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim)
        qk = qk.permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]  # make torchscript happy (cannot use tensor as tuple)
        v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            mask = mask[:, :, None] @ mask[:, None]
            mask = mask[:, None].expand(attn.shape)
            attn = attn.masked_fill(mask == 0, -6e4)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """ TNT Block"""
    def __init__(
        self,
        dim,
        in_dim,
        num_pixel,
        num_heads=12,
        in_num_head=4,
        mlp_ratio=4.,
        qkv_bias=False,
        drop=0.,
        attn_drop=0.,
        drop_path=0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        # Inner transformer
        self.norm_in = norm_layer(in_dim)
        self.attn_in = Attention(
            in_dim, in_dim, num_heads=in_num_head, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        self.norm_mlp_in = norm_layer(in_dim)
        self.mlp_in = Mlp(
            in_features=in_dim,
            hidden_features=int(in_dim * 4),
            out_features=in_dim,
            act_layer=act_layer,
            drop=drop
        )

        self.norm1_proj = norm_layer(in_dim)
        self.proj = nn.Linear(in_dim * num_pixel, dim, bias=True)
        # Outer transformer
        self.norm_out = norm_layer(dim)
        self.attn_out = Attention(
            dim, dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm_mlp = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            out_features=dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, pixel_embed, patch_embed, mask):
        # inner
        pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
        pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
        # outer
        B, N, C = patch_embed.size()
        patch_embed[:, 1:] = patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))
        # patch_embed = patch_embed + self.proj(self.norm1_proj(pixel_embed).reshape(B, N, -1))
        patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed), mask))
        patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
        return pixel_embed, patch_embed


class ImagePatchConverter(torch.nn.Module):
    def __init__(
        self,
        patch_size: int = 16,
        pixel_cnn_kernel_size: int = 7,
        max_patches: int = 512,
        min_object_value: float = 0.,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.pixel_cnn_kernel_size = pixel_cnn_kernel_size
        self.max_patches = max_patches
        n_padding = (pixel_cnn_kernel_size // 2) * 2
        self.min_object_value = min_object_value
        self.unfold = nn.Unfold(
            kernel_size=patch_size + n_padding,
            stride=patch_size,
            padding=n_padding,
        )

    @torch.no_grad()
    def __call__(self, x: torch.Tensor):
        B, C, H, W = x.shape
        NPH, NPW = H // self.patch_size, W // self.patch_size  # 縦/横方向のパッチ数

        # (B, C, H, W) -> (B, P, C * PH * PW)  ※P: パッチ数, PH,PW: パッチ幅 + パディング幅
        patches = self.unfold(x).permute(0, 2, 1)
        # C * PH * PW に対してsumを取り、大きい順にソート (物体が写ってないパッチは0になる)
        num_object_pixels, idx = patches.sum(axis=-1).sort(descending=True)
        # パッチ内に物体がどれだけ写っているのかを基にmaskを作る
        mask = (num_object_pixels > self.min_object_value).float()
        num_patches = min(int(mask.sum(dim=-1).max()), self.max_patches)

        # patchesをidxで並び替え
        patches = patches.gather(1, idx[:, :, None].expand(patches.shape))

        # 物体が写ってないサンプルをトリミング
        patches = patches[:, :num_patches]
        idx = idx[:, :num_patches]
        mask = mask[:, :num_patches]

        # 選択されたパッチに対応する座標を作成
        h_coord_idx = torch.arange(NPH)[:, None].expand(-1, NPW).reshape(-1)
        w_coord_idx = torch.arange(NPW)[None].expand(NPH, -1).reshape(-1)
        coord_idx = torch.stack([h_coord_idx, w_coord_idx], dim=-1)
        coord_idx = coord_idx[idx].to(x.device)

        patches = patches.reshape(
            B,
            -1,  # パッチ数
            C,
            self.unfold.kernel_size,
            self.unfold.kernel_size,
        )
        return dict(
            x=patches,
            mask=mask,
            coord_idx=coord_idx,
        )


class PixelEmbed(nn.Module):
    def __init__(
        self,
        patch_size=16,
        in_chans=3,
        out_chans=48,
        kernel_size=7,
    ):
        super().__init__()
        self.stride = math.ceil(kernel_size / 2)
        self.out_patch_size = math.ceil(patch_size / self.stride)
        self.out_chans = out_chans
        self.proj = nn.Conv2d(
            in_chans,
            self.out_chans,
            kernel_size=kernel_size,
            padding=0,
            stride=self.stride,
        )
        self.position = nn.Parameter(torch.zeros(
            1,
            out_chans,
            self.out_patch_size,
            self.out_patch_size,
        ))
        trunc_normal_(self.position, std=.02)

    def forward(self, x):
        B, N, C, PH, PW = x.shape
        x = x.reshape(B * N, C, PH, PW)
        x = self.proj(x)
        x = x + self.position

        # (B*N, C, H, W) -> (B, N, H*W, C) for transformer input
        x = x.reshape(B, N, self.out_chans,  self.out_patch_size ** 2)
        x = x.permute(0, 1, 3, 2)
        return x


class StaticPositionalEncoding(torch.nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.position = PositionalEncoding2D(embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.cls_token, std=.02)

    def forward(self, coord_idx):
        # Position encodingの作成
        max_h, max_w = coord_idx.reshape(-1, 2).max(dim=0)[0]
        maxlen_tensor = torch.zeros(
            (1, max_h + 1, max_w + 1, self.embed_dim),
            device=coord_idx.device
        )
        position_table = self.position(maxlen_tensor).squeeze()
        coord_idx = coord_idx.reshape(-1, 2)
        position = position_table[[coord_idx[:, 0], coord_idx[:, 1]]]
        return position


class TrainablePositionalEncoding(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_patches_h: int,
        num_patches_w: int,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches_h = num_patches_h * 2
        self.num_patches_w = num_patches_w
        self.position = nn.Parameter(torch.zeros(
            self.num_patches_h,
            self.num_patches_w,
            embed_dim,
        ))
        trunc_normal_(self.position, std=.02)

    @torch.no_grad()
    def resize(self, pretrained_embedding):
        posemb = pretrained_embedding[:, 1:]
        _, HW, D = posemb.shape
        H = W = int(math.sqrt(HW))
        posemb = posemb.reshape(1, H, W, D).permute(0, 3, 1, 2)
        import torch.nn.functional as F
        posemb = F.interpolate(
            posemb,
            size=(self.num_patches_h, self.num_patches_w),
            align_corners=False,
            mode='bilinear',
        )
        posemb = posemb.permute(0, 2, 3, 1)
        posemb = posemb.reshape(
            self.num_patches_h,
            self.num_patches_w,
            self.embed_dim
        )
        self.position[:] = nn.Parameter(posemb)

    def forward(self, coord_idx):
        max_h, max_w = coord_idx.reshape(-1, 2).max(dim=0)[0]
        assert max_h <= self.num_patches_h, max_w <= self.num_patches_w
        coord_idx = coord_idx.reshape(-1, 2)
        position = self.position[[coord_idx[:, 0], coord_idx[:, 1]]]
        return position


class PatchEmbed(torch.nn.Module):
    def __init__(
        self,
        patch_size: int,
        in_chans: int,
        embed_dim: int,
        norm_layer: torch.nn.Module,
        drop_rate: float,
        positional_encoding_type: str = "trainable",
        num_patches_h: int = 224,
        num_patches_w: int = 224,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.norm_layer = norm_layer
        self.num_pixels = patch_size ** 2
        self.embed_dim = embed_dim
        self.norm1_proj = norm_layer(self.num_pixels * in_chans)
        self.proj = nn.Linear(self.num_pixels * in_chans, embed_dim)
        self.norm2_proj = norm_layer(embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if positional_encoding_type == "trainable":
            self.position = TrainablePositionalEncoding(
                embed_dim,
                num_patches_h=num_patches_h,
                num_patches_w=num_patches_w,
            )
        elif positional_encoding_type == "static":
            self.position = StaticPositionalEncoding(embed_dim)
        else:
            raise ValueError()

    def __call__(self, x, coord_idx):
        B, N, HW, C = x.shape
        x = x.reshape(B, N, HW * C)
        x = self.norm1_proj(x)
        x = self.proj(x)
        x = self.norm2_proj(x)
        position = self.position(coord_idx).reshape(x.shape)
        x += position
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
        x = self.pos_drop(x)
        return x


@R.ModuleRegistry.add
class VariableTNT(torch.nn.Module):
    """ Transformer in Transformer - https://arxiv.org/abs/2103.00112
    """
    def __init__(
        self,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        in_dim=48,
        depth=12,
        num_heads=12,
        in_num_head=4,
        mlp_ratio=4.,
        qkv_bias=False,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        norm_layer=nn.LayerNorm,
        pixel_cnn_kernel_size=7,
        max_patches=512,
        positional_encoding_type="trainable",
        img_size_h=224,
        img_size_w=224,
        pretrained_img_size=224,
        min_object_value=0.,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_size = patch_size
        self.num_patches_h = img_size_h // patch_size
        self.num_patches_w = img_size_w // patch_size
        self.image_to_patch = ImagePatchConverter(
            patch_size=patch_size,
            pixel_cnn_kernel_size=pixel_cnn_kernel_size,
            max_patches=max_patches,
            min_object_value=min_object_value,
        )
        self.pixel_embed = PixelEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            out_chans=in_dim,
            kernel_size=pixel_cnn_kernel_size,
        )
        self.patch_embed = PatchEmbed(
            patch_size=self.pixel_embed.out_patch_size,
            in_chans=in_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
            drop_rate=drop_rate,
            num_patches_h=self.num_patches_h,
            num_patches_w=self.num_patches_w,
            positional_encoding_type=positional_encoding_type,
        )
        # Pretrained weight適用のため必要
        num_patches = (pretrained_img_size // patch_size) ** 2
        self.patch_pos = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pixel_pos = self.pixel_embed.position
        self.patch_embed.proj = self.patch_embed.proj
        self.patch_embed.norm1_proj = self.patch_embed.norm1_proj
        self.norm2_proj = self.patch_embed.norm2_proj
        self.cls_token = self.patch_embed.cls_token

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                in_dim=in_dim,
                num_pixel=self.patch_embed.num_pixels,
                num_heads=num_heads,
                in_num_head=in_num_head,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'patch_pos', 'pixel_pos', 'cls_token'}

    def forward_features(self, x):
        B, C, H, W = x.shape
        patches = self.image_to_patch(x)
        ones = torch.ones((B, 1), device=x.device)
        # 先頭に CLS token 用のマスク値を追加
        mask = torch.cat([ones, patches["mask"]], dim=1)
        pixel_embed = self.pixel_embed(patches["x"])
        patch_embed = self.patch_embed(pixel_embed, patches["coord_idx"])

        B, N, PHW, C = pixel_embed.shape
        pixel_embed = pixel_embed.reshape(B * N, PHW, C)

        for blk in self.blocks:
            pixel_embed, patch_embed = blk(pixel_embed, patch_embed, mask)

        patch_embed = self.norm(patch_embed)
        patch_embed *= mask[:, :, None]

        return dict(
            encoder_hidden_states=patch_embed,
            encoder_attention_mask=mask,
        )


@R.ModuleRegistry.add
def PretrainedVariableTNT(
    model_name: str,
    **kwargs
):
    hparams = HPARAMS[model_name]
    kwargs.update(hparams)
    model = VariableTNT(**kwargs)
    model.default_cfg = default_cfgs[model_name]
    load_pretrained(
        model,
        num_classes=model.num_classes,
        in_chans=kwargs.get("in_chans", 3),
    )
    model.patch_embed.cls_token = nn.Parameter(
        model.cls_token + model.patch_pos[:, :1]
    )
    if isinstance(model.patch_embed.position, TrainablePositionalEncoding):
        model.patch_embed.position.resize(model.patch_pos)
    return model
