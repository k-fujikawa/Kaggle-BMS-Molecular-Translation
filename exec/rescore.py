import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import click
import pandas as pd
import torch
from tqdm.auto import tqdm
from omegaconf import OmegaConf
from loguru import logger

import nncomp
import nncomp.registry as R
import nncomp_molecule
from nncomp_molecule import constants
from nncomp_molecule.generators import GeneratedInChIScorer


DEFAULT_OVERRIDES = []


@dataclass
class InferenceOption:
    in_column: str = "normed_InChI"
    stage: str = "fold=0"
    train: bool = False
    resume: Optional[bool] = None
    shuffle: bool = False
    device: Optional[int] = None
    nrows: Optional[int] = None


@click.command()
@click.argument("modeldir", type=Path)
@click.argument("input-path", type=Path)
@click.argument("overrides", nargs=-1)
def main(modeldir, input_path, overrides):
    nncomp.utils.set_seed(0)

    # Build config
    config = OmegaConf.load(modeldir / "config.yml")
    config = OmegaConf.merge(config, OmegaConf.structured(InferenceOption()))
    config = OmegaConf.merge(config, OmegaConf.from_dotlist(DEFAULT_OVERRIDES))
    OmegaConf.set_struct(config, True)
    overrides = OmegaConf.from_dotlist(overrides)
    config = OmegaConf.merge(config, overrides)

    # Load dataframe
    logger.info(f"Load: {input_path}")
    df = pd.read_csv(input_path, nrows=config.nrows)

    # For resume
    outpath = modeldir / f"{input_path.stem}.csv"
    if outpath.exists():
        if config.resume or click.confirm(f"Resume: {outpath}?"):
            generated_df = pd.read_csv(outpath, usecols=["image_id", config.in_column, "normed_score"])
            df = df.merge(generated_df, how="left")
            df = df.query("normed_score.isna()").drop(columns="normed_score")
        elif click.confirm(f"Delete: {outpath}?"):
            outpath.unlink()
        else:
            sys.exit()
    if len(df) == 0:
        sys.exit()
    logger.info(f"Output: {outpath}")

    # image_pathの付与
    if config.train:
        datadir = constants.COMPETITION_DATADIR / "train"
    else:
        datadir = constants.COMPETITION_DATADIR / "test"

    # Build dataloader
    tokenizer = R.PreprocessorRegistry.get_from_params(
        **config["tokenizer_params"]
    )
    image_transforms = R.PreprocessorRegistry.get_from_params(
        **config["image_transforms_infer"]
    )
    collate_fn = R.CollateFunctionRegistry.get_from_params(
        **config["stages"]["data_params"]["collate"],
    )

    image_paths = df.image_id.apply(
        lambda x: str(datadir / "/".join(x[:3]) / f"{x}.png")
    )
    dataset_df = df.copy().assign(
        InChI=df[config.in_column],
        image_path=image_paths,
    )
    dataset = nncomp_molecule.datasets.ImageCaptioningDataset(
        dataset=dataset_df,
        inchi_transforms=tokenizer,
        image_transforms=image_transforms,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=os.cpu_count(),
        collate_fn=collate_fn,
    )

    # Load model
    device = nncomp.utils.get_device(config.device)
    model = R.ModelRegistry.get_from_params(
        **config["model_params"]
    )
    ckpt = torch.load(modeldir / config.stage / "best.pth", map_location='cpu')
    logger.info(model.load_state_dict(ckpt))
    model = model.eval().to(device)
    scorer = GeneratedInChIScorer()

    def revert_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.tolist()
        else:
            return x

    progress = tqdm(total=len(dataloader))
    out_columns = [*df.columns, "normed_score"]
    for batch in dataloader:
        batch = nncomp.utils.to_device(batch, device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            outputs = model(**batch)
        scores = scorer(outputs["logits"], batch["next_token_ids"])
        batch_outputs_df = pd.DataFrame({
            k: revert_tensor(batch[k]) for k in out_columns
            if k in batch
        })
        batch_outputs_df["normed_score"] = scores
        batch_outputs_df = batch_outputs_df[out_columns]
        outpath.parent.mkdir(parents=True, exist_ok=True)
        batch_outputs_df.to_csv(
            outpath,
            index=False,
            header=not outpath.exists(),
            mode="a",
        )
        progress.update()


if __name__ == "__main__":
    main()
