from copy import deepcopy

import torch
import cv2
import numpy as np
import pandas as pd

import nncomp.registry as R


@R.DatasetRegistry.add
class ObjectDetectionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: pd.DataFrame,
        image_transforms: dict = {},
        length: int = None,
        eps: float = 1,
    ):
        self.dataset = dataset.to_dict("records")
        self.length = length
        self.eps = eps
        self.image_transforms = R.PreprocessorRegistry.get_from_params(
            **image_transforms,
        )

    def __len__(self):
        return self.length or len(self.dataset)

    def __getitem__(self, i):
        sample = deepcopy(self.dataset[i])
        image = cv2.imread(str(sample["image_path"]), cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype("f")
        height, width, _ = image.shape
        target = self.build_target(sample, height, width)
        sample = self.image_transforms(
            image=image,
            **target,
        )
        sample = {k: torch.Tensor(v) for k, v in sample.items()}
        return sample

    def build_target(self, sample, height, width):
        annotation_df = pd.read_csv(sample["annotation_path"])
        annotation_df = annotation_df.query("type == 'atom'").copy()
        annotation_df["x_min"] = np.clip(annotation_df.x_min, 0, width - self.eps)
        annotation_df["x_max"] = np.clip(annotation_df.x_max, self.eps, width) 
        annotation_df["y_min"] = np.clip(annotation_df.y_min, 0, height - self.eps)
        annotation_df["y_max"] = np.clip(annotation_df.y_max, self.eps, height)
        bboxes = annotation_df[["x_min", "y_min", "x_max", "y_max"]].values
        atom_types = annotation_df["label"].map(LABEL_MAPPING).values
        atom_indices = annotation_df["idx"].values
        n_Hs = annotation_df["n_Hs"].values
        return dict(
            bboxes=bboxes.astype("f"),
            atom_types=atom_types.astype("f"),
            atom_indices=atom_indices.astype("f"),
            n_Hs=n_Hs,
        )
