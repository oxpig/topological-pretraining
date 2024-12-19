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

def float_to_binary(
    array: np.ndarray,
    threshold: float = 0.5,
    below: bool = True
) -> np.ndarray:
    if below:
        return np.where(array < threshold, 1, 0)
    else:
        return np.where(array > threshold, 1, 0)

def tanimoto_filter(
    fp_1: list[DataStructs.ExplicitBitVect],
    fp_2: list[DataStructs.ExplicitBitVect],
    threshold: float = 0.5
) -> np.ndarray:
    """
    Get fingerprint filter for fp_1 based on Tanimoto similarity to fp_2.

    Parameters
    ----------
    fp_1: list[DataStructs.ExplicitBitVect]
        List of fingerprints to compare.
    fp_2: list[DataStructs.ExplicitBitVect]
        List of fingerprints to compare against.
    threshold: float
        The threshold for Tanimoto similarity. Labels values below the threshold with 1 and values
        above with 0. Default is 0.5.
    """
    out = max_tanimoto(fp_1, fp_2)
    return float_to_binary(out, threshold=threshold, below=True)

def repeat_groupkfold(
    data: np.ndarray,
    groups: np.ndarray,
    kfolds: int = 5,
    repeats: int = 1,
):
    """
    Repeat GroupKFold splits.

    Parameters
    ----------
    data: np.ndarray
        The data to split.
    groups: np.ndarray
        The groups to split the data.
    kfolds: int
        The number of folds to split the data into.
        Default is 5.
    repeats: int
        The number of times to repeat the splits.
        Default is 1.

    Returns
    -------
    out: np.ndarray
        Array of splits. Each column represents a split, where 1 is the test set and 0 is the
        train set. Each row represents a data point. Total number of splits is kfolds * repeats.
    """
    total_splits = kfolds * repeats
    out = np.zeros((data.shape[0], total_splits), dtype=int)
    for i in range(repeats):
        gkf = GroupKFold(n_splits=kfolds, shuffle=True, random_state=i)
        for j, (train_index, test_index) in enumerate(gkf.split(data, groups=groups)):
            out[test_index, i * kfolds + j] = 1
    return out

