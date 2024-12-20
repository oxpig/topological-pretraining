import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split

from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import (
    AdditionalOutput, FingeprintGenerator64, GetMorganGenerator
)
from tqdm import tqdm

from .data.datasets import BaseDataset
from .data.utils import load_dataset
from .data.mol import FPOperations, Standardizer, MorganGenerator

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
    pbar.close()
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

def batch_tanimoto_filter(
    fp_1: list[DataStructs.ExplicitBitVect],
    fp_2: list[list[DataStructs.ExplicitBitVect]],
    threshold: float = 0.5
) -> np.ndarray:
    """
    Apply Tanimoto filter to multiple sets of fingerprints.
    Useful for filtering pretraining data based on multiple benchmark sets.

    Parameters
    ----------
    fp_1: list[DataStructs.ExplicitBitVect]
        List of fingerprints to compare. (e.g. pretraining data)
    fp_2: list[list[DataStructs.ExplicitBitVect]]
        List of fingerprints to compare against. (e.g. list of benchmark data)
    threshold: float
        The threshold for Tanimoto similarity. Labels values below the threshold with 1 and values
        above with 0. Default is 0.5.

    Returns
    -------
    out: np.ndarray
        Array of filters. Each column represents a list in fp_2.
        Each row represents a data point in fp_1.
        Array is binary, where 1 indicates that the data point is dissimilar to the fp_2 set and
        0 indicates that the data point is similar to at least one molecule in fp_2 set.
        The last column represents an aggregate filter for all fp_2 sets.

    """
    out = np.zeros((len(fp_1), len(fp_2)+1))
    for i, fps in enumerate(fp_2):
        out[:, i] = tanimoto_filter(fp_1, fps, threshold=threshold)
    out[:, -1] = np.where(np.sum(out, axis=1) > 0, 1, 0)
    return out

def repeat_groupkfold(
    data: np.ndarray,
    groups: np.ndarray,
    kfolds: int = 5,
    repeats: int = 1,
    verbose: bool = True
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
    pbar = tqdm(
        total=total_splits, disable=not verbose, desc='Generating splits'
    )
    for i in range(repeats):
        gkf = GroupKFold(n_splits=kfolds, shuffle=True, random_state=i)
        for j, (train_index, test_index) in enumerate(
            gkf.split(data, groups=groups)
        ):
            out[test_index, i * kfolds + j] = 1
            pbar.update(1)
    pbar.close()
    return out

def butina_splitting(
    fps: list[DataStructs.ExplicitBitVect], threshold: float = 0.65,
    repeats: int = 1, kfolds: int = 5, verbose: bool = True
) -> np.ndarray:
    """
    Split the data using Butina clustering and GroupKFold.
    
    Parameters
    ----------
    fps: list[DataStructs.ExplicitBitVect]
        List of fingerprints to split.
    threshold: float
        The threshold for clustering.
        Default is 0.65.
    repeats: int
        The number of times to repeat the splits.
        Default is 1.
    kfolds: int
        The number of folds to split the data into.
        Default is 5.

    Returns
    -------
    out: np.ndarray
        Array of splits. Each column represents a split, where 1 is the test set and 0 is the
        train set. Each row represents a data point. Total number of splits is kfolds * repeats.
    """
    clusters = FPOperations.butina(fps, threshold=threshold, verbose=verbose)
    return repeat_groupkfold(fps, clusters, kfolds=kfolds, repeats=repeats, verbose=verbose)

def subset_indices(total: int, n: int) -> np.ndarray:
    """
    Choose a subset of indices.

    Parameters
    ----------
    total: int
        The total number of indices.
    n: int
        The number of indices to choose.

    Returns
    -------
    out: pd.DataFrame
        The subset of the data.
    """
    return np.random.choice(total, n, replace=False)
    
def indices_to_binary(indices: np.ndarray, total: int) -> np.ndarray:
    """
    Convert indices to binary array.

    Parameters
    ----------
    indices: np.ndarray
        The indices to convert.

    Returns
    -------
    out: np.ndarray
        The binary array.
    """
    out = np.zeros(total, dtype=int)
    out[indices] = 1
    return out

def preprocess(config: dict):
    """
    Preprocess the data.
    """
    pretrain_data = config['pretrain']
    data_path = config['data']
    verbose = config['verbose']
    
    pretrain_data = load_dataset(
        pretrain_data, root=data_path, compression=True,
        verbose=verbose
    )
    benchmark_data = config['benchmark']
    benchmark_data = {
        name: load_dataset(
            name, root=data_path, compression=True, verbose=verbose
        ) for name in benchmark_data
    }

    morgan_generator = MorganGenerator(**config['morgan'])
    standardizer: Standardizer = Standardizer(**config['standardizer'])
    repeats: int = config['repeats']
    kfolds: int = config['kfolds']
    butina_threshold: float = 0.65
    verbose: bool = True


    benchmark_fps = {}
    for key, value in benchmark_data.items():
        print(f'Processing {key}') if verbose else None
        smiles = value['SMILES'].tolist()
        mols = [
            standardizer(Chem.MolFromSmiles(i))
            for i in tqdm(smiles, disable=not verbose, desc='Standardizing molecules')
        ]
        # add mol checker
        fps = [
            morgan_generator.dense(i)
            for i in tqdm(mols, disable=not verbose, desc='Generating fingerprints')
        ]
        butina_splits = butina_splitting(
            value, threshold=butina_threshold, repeats=repeats,
            kfolds=kfolds, verbose=verbose
        )
        butina_splits = pd.DataFrame(
            butina_splits, columns=[f'split_{i}'
            for i in range(butina_splits.shape[1])
        ])
        benchmark_data[key].join(butina_splits)
        benchmark_data[key].save()
        benchmark_fps[key] = fps

    pretrain_smiles = pretrain_data['SMILES'].tolist()
    pretrain_mols = [standardizer(Chem.MolFromSmiles(i)) for i in pretrain_smiles]
    
    pretrain_fps = [morgan_generator.dense(i) for i in pretrain_mols]
    pretrain_filter = batch_tanimoto_filter(
        pretrain_fps, benchmark_fps.values(), threshold=0.5
    )
    num_keep = np.sum(pretrain_filter[:, -1])
    random_indices = subset_indices(pretrain_data.shape[0], num_keep)
    random_indices = indices_to_binary(random_indices, pretrain_data.shape[0])
    pretrain_filter = np.concatenate([pretrain_filter, ], axis=1)
    
    cols = [f'{key}_filter' for key in benchmark_fps.keys()] + ['aggregate', 'random']
    pretrain_filter = pd.DataFrame(pretrain_filter, columns=cols)
    pretrain_data.join(pretrain_filter)
    pretrain_data.save()
