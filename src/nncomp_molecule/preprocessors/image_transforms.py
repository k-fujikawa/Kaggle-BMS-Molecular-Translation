import math
import albumentations
from typing import List

import cv2
import torch
import numpy as np
from torchvision.transforms import ToPILImage, Resize, InterpolationMode
from PIL import Image

import nncomp.registry as R


@R.PreprocessorRegistry.add
class SaltAndPepperNoise(albumentations.ImageOnlyTransform):
    def __init__(self, p=1.0, always_apply=False):
        super().__init__(always_apply, p)

    def apply(self, image, **params):
        salt_amount = np.random.uniform(0, 0.3)
        salt = np.random.uniform(0, 1, image.shape[:2]) < salt_amount
        image = np.where(salt[:, :, None], np.zeros_like(image), image)
        pepper_amount = np.random.uniform(0, 0.001)
        pepper = np.random.uniform(0, 1, image.shape[:2]) < pepper_amount
        image = np.where(pepper[:, :, None], np.ones_like(image), image)
        return image


@R.PreprocessorRegistry.add
class Binalize(albumentations.ImageOnlyTransform):
    def __init__(self, p=1.0, always_apply=False):
        super().__init__(always_apply, p)

    def apply(self, image, **params):
        return (~(image == 255)).astype("f")


@R.PreprocessorRegistry.add
class PILResize(albumentations.ImageOnlyTransform):
    def __init__(self, height, width, p=1.0, always_apply=False):
        super().__init__(always_apply, p)
        self.to_pil = ToPILImage()
        self.resize = Resize(
            size=(width, height),
            # interpolation=InterpolationMode.BICUBIC,
        )

    def apply(self, image, **params):
        image = self.to_pil(image)
        image = self.resize(image)
        image = np.array(image)[:, :, None]
        return image


@R.PreprocessorRegistry.add
class Denoise(albumentations.ImageOnlyTransform):
    def __init__(
        self,
        kernel_size=3,
        thresh=1,
        p=1.0,
        always_apply=False
    ):
        super().__init__(always_apply, p)
        padding = kernel_size // 2
        self.conv = torch.nn.Conv2d(
            1, 1,
            (kernel_size, kernel_size),
            padding=padding,
            bias=False,
        )
        self.conv.weight = torch.nn.Parameter(
            torch.ones((1, 1, kernel_size, kernel_size))
        )
        self.conv.require_grad = False
        self.thresh = thresh

    @torch.no_grad()
    def apply(self, image, **params):
        denoised = image.reshape(-1).copy()
        image = torch.Tensor(image.transpose(2, 0, 1))
        mask = self.conv(image[None]).detach().numpy() <= self.thresh
        denoised[mask.squeeze().reshape(-1)] = 0
        denoised = denoised.reshape(image.shape)
        return denoised.transpose(1, 2, 0)


@R.PreprocessorRegistry.add
class Rescale(albumentations.ImageOnlyTransform):
    def __init__(self, h_scale, w_scale, p=1.0, always_apply=False):
        super().__init__(always_apply, p)
        self.h_scale = h_scale
        self.w_scale = w_scale

    def apply(self, image, **params):
        h, w = image.shape[:2]
        rescaled_image = cv2.resize(
            image,
            (int(w * self.w_scale), int(h * self.h_scale)),
        )
        return rescaled_image


@R.PreprocessorRegistry.add
class RescaleToArea(albumentations.ImageOnlyTransform):
    def __init__(self, base_size, p=1.0, always_apply=False):
        super().__init__(always_apply, p)
        self.base_size = base_size
        self.target_area = base_size ** 2

    def apply(self, image, **params):
        h, w = image.shape[:2]
        new_w = int(np.sqrt(self.target_area / (h / w)))
        new_h = int(np.sqrt(self.target_area / (w / h)))
        rescaled_image = cv2.resize(
            image,
            (new_w, new_h),
        )
        return rescaled_image


def pad_image_outsides_dhw(image, h, w, pad_value=0):
    d, h_origin, w_origin = image.shape
    padded_image = torch.full(
        (d, h, w),
        pad_value,
        dtype=image.dtype
    )
    center_h_origin, center_h_padded = h_origin // 2, h // 2
    center_w_origin, center_w_padded = w_origin // 2, w // 2
    h0 = center_h_padded - center_h_origin
    w0 = center_w_padded - center_w_origin
    padded_image[:, h0:h0 + h_origin, w0:w0 + w_origin] = image
    return padded_image


@R.PreprocessorRegistry.add
class ImageInsideCropping(albumentations.ImageOnlyTransform):
    def apply(self, image, **params):
        row_indices, col_indices, _ = list(np.where(image > 0))
        min_x, max_x = col_indices.min(), col_indices.max()
        min_y, max_y = row_indices.min(), row_indices.max()
        cropped_image = image[min_y:max_y, min_x:max_x]
        return cropped_image


@R.PreprocessorRegistry.add
class ImageOutsidePadding(albumentations.ImageOnlyTransform):
    def __init__(self, max_h, max_w, pad_value=0, p=1.0, always_apply=False):
        super().__init__(always_apply, p)
        self.max_h = max_h
        self.max_w = max_w
        self.pad_value = pad_value

    def apply(self, image, **params):
        d, h, w = image.shape
        assert h <= self.max_h, w <= self.max_w
        padded_image = pad_image_outsides_dhw(
            image=image,
            h=self.max_h,
            w=self.max_w,
            pad_value=self.pad_value,
        )
        return padded_image


@R.CollateFunctionRegistry.add
class ImageOutsidePaddingCollateFunction:
    def __init__(self, patch_size=32, pad_value=0):
        self.patch_size = patch_size
        self.pad_value = pad_value

    def __call__(self, batch: List):
        h_max = max([x.shape[1] for x in batch])
        h_max = math.ceil(h_max / self.patch_size) * self.patch_size
        w_max = max([x.shape[2] for x in batch])
        w_max = math.ceil(w_max / self.patch_size) * self.patch_size

        batch = torch.stack(
            [
                pad_image_outsides_dhw(
                    image=x,
                    h=h_max,
                    w=w_max,
                    pad_value=self.pad_value,
                )
                for x in batch
            ]
        )
        return batch
