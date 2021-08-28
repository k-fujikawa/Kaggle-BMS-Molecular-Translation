import os

import click
import nncomp.io.download
from loguru import logger


KAGGLE_COMPETITIONS = [
    "bms-molecular-translation"
]
KAGGLE_DATASETS = [
    "kfujikawa/bms-preprocess-v2",
    "kfujikawa/bms-tokenizers-v1",
    "kfujikawa/bms-kfold",
    "kfujikawa/kf-bms-candidates-v2",
    "kfujikawa/bms-preprocess-with-pseudo-lb074",
]
KAGGLE_KERNEL_OUTPUTS = []


@click.command()
@click.option("--overwrite", is_flag=True)
@click.argument("datadir", default=os.environ["INPUTDIR"])
def main(datadir, overwrite):
    for i, competition_id in enumerate(KAGGLE_COMPETITIONS):
        logger.info(
            f"Download Kaggle kernel outputs: {competition_id}"
            f" ({i+1} / {len(KAGGLE_COMPETITIONS)})"
        )
        nncomp.io.download.download_from_kaggle_competition(
            competition_id,
            f"{datadir}/{competition_id}",
            overwrite=overwrite,
        )

    for i, kernel_id in enumerate(KAGGLE_KERNEL_OUTPUTS):
        logger.info(
            f"Download Kaggle kernel outputs: {kernel_id}"
            f" ({i+1} / {len(KAGGLE_KERNEL_OUTPUTS)})"
        )
        nncomp.io.download.download_from_kaggle_kernel_output(
            kernel_id,
            f"{datadir}/{kernel_id}",
            decompress_exclude_suffixes=[".json"],
            overwrite=overwrite,
        )

    for i, dataset_id in enumerate(KAGGLE_DATASETS):
        logger.info(
            f"Download Kaggle datasets: {dataset_id}"
            f" ({i+1} / {len(KAGGLE_DATASETS)})"
        )
        nncomp.io.download.download_from_kaggle_dataset(
            dataset_id,
            f"{datadir}/{dataset_id}",
            decompress_exclude_suffixes=[".json"],
            overwrite=overwrite,
        )


if __name__ == '__main__':
    main()
