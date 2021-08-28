import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import click
import numpy as np
import torch
import pandas as pd
import Levenshtein
from tqdm.auto import tqdm
from loguru import logger
from omegaconf import OmegaConf

import nncomp
import nncomp.registry as R
import nncomp_molecule


PREPROCESSED_DIR = nncomp.constants.INPUTDIR / "kfujikawa/bms-preprocess-v2"
KFOLD_FILEPATH = nncomp.constants.INPUTDIR / "kfujikawa/bms-kfold/10fold.csv"
DEFAULT_OVERRIDES = [
    "stages.data_params.loaders_params.train.dataset.image_transforms=${image_transforms_infer}",  # NOQA
    "stages.data_params.loaders_params.train.drop_last=false",
    "stages.data_params.loaders_params.train.shuffle=false",
]


@dataclass
class InferenceOption:
    num_beams: int = 1
    dataset: str = "valid"
    stage: str = "fold=0"
    shuffle: bool = False
    device: Optional[int] = None


@click.command()
@click.argument("modeldir", type=Path)
@click.argument("overrides", nargs=-1)
def main(modeldir, overrides):
    nncomp.utils.set_seed(0)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # Build config
    config = OmegaConf.load(modeldir / "config.yml")
    config = OmegaConf.merge(config, OmegaConf.structured(InferenceOption()))
    config = OmegaConf.merge(config, OmegaConf.from_dotlist(DEFAULT_OVERRIDES))
    OmegaConf.set_struct(config, True)
    overrides = OmegaConf.from_dotlist(overrides)
    config = OmegaConf.merge(config, overrides)

    # For debug
    outpath = modeldir / f"{config.dataset}_beam={config.num_beams}.csv"
    if config.debug:
        outpath = Path(outpath).with_suffix(".debug.csv")
        for key, path in config.inputs.items():
            config.inputs[key] = str(Path(path).with_suffix(".debug.pkl"))
    print(OmegaConf.to_yaml(config))

    # Build experiment
    experiment = R.ExperimentRegistry.get_from_params(
        **config.experiment_params,
        config=OmegaConf.to_container(config, resolve=True),
    )

    # Load dataframe
    df = experiment.get_rawdata(config.stage, config.dataset)
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

    # Load model
    device = nncomp.utils.get_device(config.device)
    model = R.ModelRegistry.get_from_params(
        **config.model_params
    )
    ckpt = torch.load(modeldir / config.stage / "best.pth", map_location='cpu')
    logger.info(model.load_state_dict(ckpt))
    model = model.eval().to(device)

    # Build generator
    generation_config = nncomp_molecule.generators.GenerationConfig(
        num_beams=config.num_beams,
        num_return_sequences=config.num_beams,
        use_cache=True,
    )
    generator = nncomp_molecule.generators.EnsenmbleBeamSearchGenerator(
        config=generation_config,
        tokenizer=model.tokenizer,
        models=[model],
        weights=[1],
    )

    # Inference
    logger.info("Start to inference")
    outputs_df = pd.DataFrame()
    loader = experiment.get_loader(df, config.stage, config.dataset)
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
