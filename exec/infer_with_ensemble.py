import os
import sys
import yaml
from pathlib import Path

import click
import torch
import numpy as np
import pandas as pd
import Levenshtein
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from loguru import logger

import nncomp
import nncomp.registry as R
import nncomp_molecule  # NOQA
from nncomp_molecule import constants
from nncomp_molecule.preprocessors import (
    normalize_inchi_batch,
    disable_rdlogger,
)


PREPROCESSED_DIR = nncomp.constants.INPUTDIR / "kfujikawa/bms-preprocess-v2"
KFOLD_FILEPATH = nncomp.constants.INPUTDIR / "kfujikawa/bms-kfold/10fold.csv"


def load_dataset(dataset, stage, debug):
    suffix = ".debug.pkl" if debug else ".pkl"
    data_type = "test" if dataset == "test" else "train"
    fold = stage.split("=")[-1]
    filepath = PREPROCESSED_DIR / f"{data_type}{suffix}"
    logger.info(f"Load: {filepath}")
    df = pd.read_pickle(filepath)
    kfold_df = pd.read_csv(KFOLD_FILEPATH)
    if dataset == "test":
        return df
    elif dataset == "valid":
        df = df.merge(kfold_df, on="image_id").query("fold == @fold")
        return df
    elif dataset == "train":
        df = df.merge(kfold_df, on="image_id").query("fold != @fold")
        return df


@click.command()
@click.argument("config", type=Path)
@click.argument("overrides", nargs=-1)
def main(config, overrides):
    nncomp.utils.set_seed(0)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # Build config
    config = nncomp.config.load_config(config)
    OmegaConf.set_struct(config, True)
    overrides = OmegaConf.from_dotlist(overrides)
    config = OmegaConf.merge(config, overrides)

    # For debug
    if config.debug:
        config.name = "debug:infer"
    outpath = Path(config.outdir) / f"{config.dataset}_beam={config.num_beams}.csv"
    print(OmegaConf.to_yaml(config))

    # Load dataframe
    df = load_dataset(config.dataset, config.stage, config.debug)
    if config.shuffle:
        df = df.sample(frac=1)

    # For resume
    if outpath.exists():
        if click.confirm(f"Resume: {outpath}?"):
            generated_df = pd.read_csv(outpath, usecols=["image_id"])
            df = df.query("~image_id.isin(@generated_df.image_id)")
        elif click.confirm(f"Delete: {outpath}?"):
            outpath.unlink()
        else:
            sys.exit()
    logger.info(f"Output: {outpath}")

    # Build dataloader
    tokenizer = R.PreprocessorRegistry.get_from_params(
        **config.tokenizer_params
    )
    collate_fn = R.CollateFunctionRegistry.get_from_params(
        **config.collate
    )
    dataset = nncomp_molecule.datasets.ImageCaptioningDataset(
        dataset=df,
        inchi_transforms=tokenizer,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=os.cpu_count(),
        collate_fn=collate_fn,
    )

    # Mkdir / copy config
    outpath.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, outpath.parent / "ensemble_config.yml")

    # Load models
    models, weights, image_transforms = [], [], []
    for modeldir, weight in config.models.items():
        modeldir = Path(modeldir)
        model_config = OmegaConf.load(modeldir.parent / "config.yml")
        assert model_config.tokenizer_params == config.tokenizer_params
        model = R.ModelRegistry.get_from_params(
            **model_config.model_params
        )
        image_transform = R.PreprocessorRegistry.get_from_params(
            **model_config.image_transforms_infer
        )
        ckpt = torch.load(modeldir / "best.pth", map_location='cpu')
        logger.info(model.load_state_dict(ckpt))
        models.append(model)
        weights.append(weight)
        image_transforms.append(image_transform)

    # Build generator
    device = nncomp.utils.get_device(config.device)
    generation_config = nncomp_molecule.generators.GenerationConfig(
        num_beams=config.num_beams,
        num_return_sequences=config.num_beams,
        use_cache=True,
    )
    generator = nncomp_molecule.generators.EnsenmbleBeamSearchGenerator(
        config=generation_config,
        tokenizer=tokenizer,
        models=models,
        weights=weights,
        transforms=image_transforms,
    )
    generator = generator.eval().to(device)

    # Inference
    logger.info("Start to inference")
    outputs_df = pd.DataFrame()
    progress = tqdm(total=len(loader))
    for batch in loader:
        if "InChI" in batch:
            batch["InChI_GT"] = batch["InChI"]
        batch_outputs_df = pd.DataFrame({
            k: np.repeat(
                batch[k],
                generation_config.num_return_sequences,
                axis=0,
            )
            for k in ["image_id", "InChI", "InChI_GT"]
            if k in batch
        })
        batch = nncomp.utils.to_device(batch, device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            input_ids = torch.full(
                (len(batch["image_id"]), 1),
                model.tokenizer.token_to_id("<BOS>"),
                device=device,
            )
            encoder_outputs = generator.encode(batch["image"])
            outputs = generator.generate(
                input_ids=input_ids,
                encoder_outputs=encoder_outputs,
            )
            outputs = generator.postprocess(outputs)
            normed_score = generator.rescore(
                InChIs=outputs.normed_InChI,
                encoder_outputs=encoder_outputs,
            )

        batch_outputs_df["score"] = outputs.log_likelihood
        batch_outputs_df["InChI"] = outputs.InChI
        batch_outputs_df["is_valid"] = outputs.is_valid
        batch_outputs_df["normed_InChI"] = outputs.normed_InChI
        batch_outputs_df["normed_score"] = normed_score

        if "InChI_GT" in batch:
            batch_outputs_df["levenshtein"] = [
                Levenshtein.distance(InChI, InChI_GT)
                for InChI, InChI_GT
                in batch_outputs_df[["normed_InChI", "InChI_GT"]].values
            ]
            outputs_df = outputs_df.append(
                batch_outputs_df[["image_id", "levenshtein"]],
                ignore_index=True,
            )
            top1_outputs_df = outputs_df.groupby("image_id").first()
            progress.set_postfix(
                levenshtein=top1_outputs_df.levenshtein.mean()
            )

        # Write outputs
        progress.update()
        batch_outputs_df.to_csv(
            outpath,
            index=False,
            header=not outpath.exists(),
            mode="a",
        )


if __name__ == "__main__":
    main()
