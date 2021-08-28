from copy import deepcopy
from typing import Union, Callable

import albumentations as A
import torch
import cv2
import numpy as np
import pandas as pd

import nncomp.registry as R


@R.DatasetRegistry.add
class ImageCaptioningDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: pd.DataFrame,
        image_transforms: Union[dict, Callable] = None,
        inchi_transforms: Union[dict, Callable] = None,
        in_chans: int = 1,
        length: int = None,
    ):
        self.dataset = dataset.to_dict("records")
        self.length = length
        self.image_transforms = image_transforms
        self.in_chans = in_chans
        if isinstance(image_transforms, dict):
            self.image_transforms = R.PreprocessorRegistry.get_from_params(
                **image_transforms,
            )
        self.inchi_transforms = inchi_transforms
        if isinstance(inchi_transforms, dict):
            self.inchi_transforms = R.PreprocessorRegistry.get_from_params(
                **inchi_transforms,
            )

    def __len__(self):
        return self.length or len(self.dataset)

    def __getitem__(self, i):
        sample = deepcopy(self.dataset[i])
        sample.update(self.transform_image(sample))
        if self.inchi_transforms is not None:
            sample.update(self.transform_inchi(sample))
        return sample

    def transform_inchi(self, sample):
        outputs = {}
        if "token_ids" in sample:
            outputs["token_ids"] = sample["token_ids"]
            outputs["next_token_ids"] = np.append(sample["token_ids"][1:], 0)
        elif "InChI" in sample:
            outputs = self.inchi_transforms(sample["InChI"])
        return outputs

    def transform_image(self, sample):
        image = cv2.imread(sample["image_path"], cv2.IMREAD_GRAYSCALE)
        if self.in_chans == 1:
            image = image[:, :, None]
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype("f")
        h, w = image.shape[:2]
        if h > w:
            image = np.flipud(image.transpose(1, 0, 2))
        if self.image_transforms is not None:
            image = self.image_transforms(image=image)["image"]
        return dict(
            image=image
        )
