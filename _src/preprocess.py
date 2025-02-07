import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split, StratifiedGroupKFold

from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import (
    AdditionalOutput, GetMorganGenerator
)
from tqdm import tqdm

from .data.datasets import BaseDataset
from .data.utils import load_dataset, load_molecules
from .data.mol import FPOps, Standardizer, MorganGenerator

def max_tanimoto(
    fps_1: list[DataStructs.ExplicitBitVect],
    fps_2: list[DataStructs.ExplicitBitVect],
    verbose: bool = False,
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
        sims = FPOps.bulk_tanimoto(fp_1, fps_2)
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
    threshold: float = 0.5,
    verbose: bool = False
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
    out = max_tanimoto(fp_1, fp_2, verbose=verbose)
    return float_to_binary(out, threshold=threshold, below=True)

def batch_tanimoto_filter(
    fp_1: list[DataStructs.ExplicitBitVect],
    fp_2: list[list[DataStructs.ExplicitBitVect]],
    threshold: float = 0.5,
    verbose: bool = False,
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
    pbar = tqdm(fp_2, disable=not verbose, desc='Benchmark dataset')
    for i, fps in enumerate(fp_2):
        out[:, i] = tanimoto_filter(fp_1, fps, threshold=threshold, verbose=verbose)
        pbar.update(1)
    pbar.close()
    out[:, -1] = np.where(np.sum(out[:,:-1], axis=1) == len(fp_2), 1, 0)
    return out

def batch_max_tanimoto(
    fp_1: list[DataStructs.ExplicitBitVect],
    fp_2: list[list[DataStructs.ExplicitBitVect]],
    verbose: bool = False,
) -> np.ndarray:
    out = np.zeros((len(fp_1), len(fp_2)))
    pbar = tqdm(total=len(fp_2), disable=not verbose)
    for i, fps in enumerate(fp_2):
        out[:, i] = max_tanimoto(fp_1, fps, verbose=verbose)
        pbar.update(1)
    pbar.close()

    return out
def repeat_groupkfold(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray|None = None,
    kfolds: int = 5,
    repeats: int = 1,
    verbose: bool = True,
    stratified: bool = False,
):
    """
    Repeat GroupKFold splits.

    Parameters
    ----------
    X: np.ndarray
        data to split.
    y: np.ndarry
        data labels to split.
    groups: 
        The groups to split the data.
    kfolds: int
        The number of folds to split the data into.
        Default is 5.
    repeats: int
        The number of times to repeat the splits.
        Default is 1.
    stratified: bool
        Whether to stratify splits by y. Only applicable
        for classification.

    Returns
    -------
    out: np.ndarray
        Array of splits. Each column represents a split, where 1 is the test set and 0 is the
        train set. Each row represents a data point. Total number of splits is kfolds * repeats.
    """
    total_splits = kfolds * repeats
    out = np.full((len(X), total_splits), fill_value='Train')
    pbar = tqdm(
        total=total_splits, disable=not verbose, desc='Generating splits'
    )
    gkf = StratifiedGroupKFold if stratified else GroupKFold
    for i in range(repeats):
        splitter = gkf(n_splits=kfolds, shuffle=True, random_state=i)
        for j, (train_index, test_index) in enumerate(
            splitter.split(X, y, groups=groups)
        ):
            out[test_index, i * kfolds + j] = 'Test'
            pbar.update(1)
    pbar.close()
    return out

def butina_splitting(
    fps: list[DataStructs.ExplicitBitVect], y: np.ndarray = None, threshold: float = 0.65,
    repeats: int = 1, kfolds: int = 5, verbose: bool = True,
    stratified = False
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
    stratified: bool
        Whether to stratify splitting by y. Only applicable for classification.
        Default is False.

    Returns
    -------
    out: np.ndarray
        Array of splits. Each column represents a split, where 1 is the test set and 0 is the
        train set. Each row represents a data point. Total number of splits is kfolds * repeats.
    """
    clusters = FPOps.butina(fps, threshold=threshold, verbose=verbose)
    splits = repeat_groupkfold(
        fps, y=y, groups=clusters,
        kfolds=kfolds, repeats=repeats, verbose=verbose,
        stratified=stratified
    )
    return splits, clusters

def subset_indices(total: int|np.ndarray, n: int) -> np.ndarray:
    """
    Choose a random subset of indices.

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

def splitters(name: str):
    """
    Get the splitter function by name.
    """
    splitters = {
        'butina': butina_splitting
    }
    return splitters[name]

def preprocess(config: dict):
    """
    Preprocess the data.
    """
    data_path = config['data']
    verbose: bool = config['verbose']

    benchmark_data = config['benchmark']

    morgan_generator = MorganGenerator(verbose=verbose, **config['morgan'])
    standardizer: Standardizer = Standardizer(**config['standardizer'])
    splitter_kind = config['splitting']['kind']
    splitter = splitters(splitter_kind)
    splitter_params = config['splitting']['params']

    benchmark_fps = {}
    # fps as explicitbitvect
    morgan_generator.asarray = False

    for benchmark in benchmark_data:
        print(f'Processing {benchmark}') if verbose else None
        df: BaseDataset = load_dataset(
            benchmark, root=data_path, compression=True,
            verbose=verbose, standardizer=standardizer
        )
        
        rdkit_passes = df[df['rdkit_pass'] == True]
        mols = df.rdkit_mols[rdkit_passes.index]
        
        fps = morgan_generator(mols)
        if 'split' not in df.columns[-1]:
            print(f'Generating splits for {benchmark}') if verbose else None
            splits, groups = splitter(
                fps, y=rdkit_passes.y.values, verbose=verbose, **splitter_params
            )
            if groups is not None:
                df[f'{splitter_kind}_cluster'] = np.nan
                df.loc[rdkit_passes.index, f'{splitter_kind}_cluster'] = groups

            splits = pd.DataFrame(
                splits,
                columns=[f'split_{i}' for i in range(splits.shape[1])],
                index=rdkit_passes.index
            )
            csv_path = df.csv
            df = df.join(splits)
            df.to_csv(csv_path, compression='infer', index=False)
        else:
            print(f'Splits already exist for {benchmark}') if verbose else None
        benchmark_fps[benchmark] = fps
    pretrain_data = config['pretrain']
    pretrain_data: BaseDataset = load_dataset(
        pretrain_data, root=data_path, compression=True,
        verbose=verbose
    )

    if 'butina_filter' and 'max_tanimoto' not in pretrain_data.columns:
        print(f'Processing filters for {pretrain_data.name}') if verbose else None
        print(f'Data shape: {pretrain_data.shape}') if verbose else None
        rdkit_passes = pretrain_data[pretrain_data['rdkit_pass'] == True]
        rdkit_fails = pretrain_data[pretrain_data['rdkit_pass'] == False]
        pretrain_mols = pretrain_data.rdkit_mols[rdkit_passes.index]
        
        # fps as explicitbitvect
        pretrain_fps = morgan_generator(pretrain_mols)

        
        max_tanimote_scores = batch_max_tanimoto(
            pretrain_fps, benchmark_fps.values(), verbose=verbose
        )
        max_tanimote_scores_all = np.max(max_tanimote_scores, axis=1)
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        num_keep = None
        for threshold in thresholds:
            pretrain_filter = float_to_binary(max_tanimote_scores_all, threshold=threshold, below=True)
            for fail in rdkit_fails.index:
                pretrain_filter = np.insert(pretrain_filter, fail, 0, axis=0)
            if num_keep is None:
                num_keep = int(np.sum(pretrain_filter))
            else:
                filter_indices = np.where(pretrain_filter == 1)[0]
                filter_indices = subset_indices(filter_indices, num_keep)
                pretrain_filter = indices_to_binary(filter_indices, len(pretrain_filter))
                
            pretrain_data[f'butina_filter_{threshold}'] = pretrain_filter

        for fail in rdkit_fails.index:
            max_tanimote_scores_all = np.insert(max_tanimote_scores_all, fail, np.nan, axis=0)
            max_tanimote_scores = np.insert(max_tanimote_scores, fail, np.nan, axis=0)
        pretrain_data['max_tanimoto'] = max_tanimote_scores_all
        for i, benchmark in enumerate(benchmark_fps.keys()):
            pretrain_data[f'max_tanimoto_{benchmark}'] = max_tanimote_scores[:, i]

        pretrain_data.save()
    else:
        print(f'Filters already exist for {pretrain_data.name}') if verbose else None
