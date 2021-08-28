import numpy as np
import pandas as pd
from torch.utils.data.sampler import BatchSampler

import nncomp.registry as R


@R.SamplerRegistry.add
class BMSBalanceClassSampler(BatchSampler):
    def __init__(
        self,
        dataset: pd.DataFrame,
        class_name: str,
        samples_per_batch: dict,
        samples_per_epoch: int = 1_000_000,
    ):
        self.samples_per_epoch = samples_per_epoch
        self.class_values = dataset[class_name].unique()
        self.samples_per_batch = samples_per_batch
        self.batch_size = sum(self.samples_per_batch.values())
        labels = dataset[class_name]
        self.lbl2idx = {
            str(label): np.arange(len(labels))[labels == label].tolist()
            for label in self.class_values
        }
        self.n_samples = [len(x) for x in self.lbl2idx.values()]
        self.length = samples_per_epoch // self.batch_size

    def __iter__(self):
        for i in range(len(self)):
            indices = []
            for key in sorted(self.lbl2idx):
                indices += np.random.choice(
                    self.lbl2idx[key], self.samples_per_batch[key],
                ).tolist()
            yield indices

    def __len__(self):
        return self.length
