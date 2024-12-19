import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split

from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import (
    AdditionalOutput, FingeprintGenerator64, GetMorganGenerator
)
from tqdm import tqdm

from .data.datasets import BaseDataset
from .data.mol import FPOperations, Standardizer

def max_tanimoto(
    fps_1: list[DataStructs.ExplicitBitVect],
    fps_2: list[DataStructs.ExplicitBitVect],
    verbose: bool = True,
    ) -> np.ndarray:
    """
    Calculate the maximum Tanimoto similarity for each molecule in fps_1 to all molecules in fps_2.

    Parameters
    ----------
    fps_1: list[DataStructs.ExplicitBitVect]
        List of fingerprints to compare.
    fps_2: list[DataStructs.ExplicitBitVect]
        List of fingerprints to compare against.
    verbose: bool
        Whether to show the progress bar or not.
        Default is True.
    """
    out = np.zeros((len(fps_1)))
    pbar = tqdm(total=len(fps_1), disable=not verbose)
    for i, fp_1 in enumerate(fps_1):
        sims = FPOperations.bulk_tanimoto(fp_1, fps_2)
        out[i] = np.max(sims)
        pbar.update(1)
    return out
