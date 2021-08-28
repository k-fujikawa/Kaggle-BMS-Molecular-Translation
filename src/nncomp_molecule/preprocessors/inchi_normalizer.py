import os
from typing import List
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import rdkit.RDLogger
from rdkit import Chem
from tqdm.auto import tqdm
from loguru import logger


def normalize_inchi(inchi: str):
    try:
        mol = Chem.MolFromInchi(inchi)
        if mol is not None:
            return Chem.MolToInchi(mol)
    except Exception:
        return None


def _normalize_inchi_batch(inchis: List[str], verbose: bool = True):
    results = []
    executor = ProcessPoolExecutor(max_workers=1)
    if verbose:
        logger.info("Start to normalize InChI")
    for inchi in tqdm(inchis, disable=not verbose):
        try:
            results.append(
                executor.submit(normalize_inchi, inchi).result()
            )
        except Exception as e:
            if isinstance(e, KeyboardInterrupt):
                raise KeyboardInterrupt()
            results.append(None)
            executor.shutdown()
            executor = ProcessPoolExecutor(max_workers=1)
    executor.shutdown()
    return pd.Series(results, name="InChI")


def normalize_inchi_batch(
    inchis: List[str],
    n_workers: int = os.cpu_count(),
    verbose: bool = True,
):
    if n_workers <= 1:
        return _normalize_inchi_batch(inchis, verbose)
    groups = np.array_split(inchis, n_workers)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(
                _normalize_inchi_batch,
                inchis=group,
                verbose=verbose
            )
            for group in groups
        ]
        normed_inchis = pd.concat(
            [f.result() for f in futures],
            ignore_index=True
        )
    return normed_inchis


def disable_rdlogger():
    rdlogger = rdkit.RDLogger.logger()
    rdlogger.setLevel(rdkit.RDLogger.ERROR)
    rdkit.rdBase.DisableLog('rdApp.error')
