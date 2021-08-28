import pickle
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import click
import numpy as np
import pandas as pd
import rdkit
from loguru import logger
from tqdm.auto import tqdm
from PIL import Image

import nncomp
from nncomp_molecule.preprocessors import MolecularBBoxGenerator


rdlogger = rdkit.RDLogger.logger()
rdlogger.setLevel(rdkit.RDLogger.ERROR)
rdkit.rdBase.DisableLog('rdApp.error')


def generate_annotations(row: dict):
    image_id, outdir = row["image_id"], row["outdir"]
    image_outdir = \
        outdir / "images" / f"{'/'.join(image_id[:3])}"
    noised_image_outdir = \
        outdir / "noised_images" / f"{'/'.join(image_id[:3])}"
    svg_outdir = \
        outdir / "svgs" / f"{'/'.join(image_id[:3])}"
    annotation_outdir = \
        outdir / "annotations" / f"{'/'.join(image_id[:3])}"

    generator = MolecularBBoxGenerator()
    image, noised_image, svg, annotation_df = generator(row["InChI"])
    if image is None:
        return {}

    annotation_df["image_id"] = image_id
    image_outdir.mkdir(exist_ok=True, parents=True)
    noised_image_outdir.mkdir(exist_ok=True, parents=True)
    svg_outdir.mkdir(exist_ok=True, parents=True)
    annotation_outdir.mkdir(exist_ok=True, parents=True)

    # Save image
    Image.fromarray(image).save(image_outdir / f"{image_id}.png")
    # Save noised image
    Image.fromarray(noised_image).save(noised_image_outdir / f"{image_id}.png")
    # Save SVG
    with open(svg_outdir / f"{image_id}.svg", "wb") as f:
        f.write(svg)

    # Save annotations
    annotation_df.to_csv(annotation_outdir / f"{image_id}.csv", index=False)

    return dict(
        image_id=image_id,
        InChI=row["InChI"],
    )


def generate_annotations_with_process(rows: list):
    results = []
    executor = ProcessPoolExecutor(max_workers=1)
    for row in tqdm(rows):
        try:
            results.append(executor.submit(generate_annotations, row).result())
        except Exception as e:
            if isinstance(e, KeyboardInterrupt):
                raise KeyboardInterrupt()
            executor.shutdown()
            executor = ProcessPoolExecutor(max_workers=1)
    executor.shutdown()
    return pd.DataFrame(results).dropna()


@click.command()
@click.argument("input", type=Path)
@click.argument("outdir", type=Path)
@click.option("--n-rows", type=int, default=None)
@click.option("--n-workers", type=int, default=mp.cpu_count())
def main(input, outdir, n_rows, n_workers):
    if outdir.exists() and not click.confirm(f"Overwrite: {outdir}?"):
        sys.exit()

    nncomp.utils.set_seed(0)
    records = pd.read_csv(input, nrows=n_rows)\
        .assign(outdir=outdir)\
        .to_dict("records")

    groups = np.array_split(records, n_workers)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(generate_annotations_with_process, group)
            for group in tqdm(groups)
        ]
        results_df = pd.concat(
            [f.result() for f in futures],
            ignore_index=True
        )

    results_df.to_csv(outdir / "datasets.csv", index=False)
    results_df.to_pickle(outdir / "datasets.pkl")
    results_df.head(1000).to_pickle(outdir / "datasets.debug.pkl")


if __name__ == '__main__':
    main()
