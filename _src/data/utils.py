from . import datasets
from .mol import Standardizer

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from tqdm import tqdm
from typing import Literal

def load_dataset(
    name: str,
    root: str|None = None,
    compression: bool = True,
    verbose: bool = False,
    standardizer: Standardizer = Standardizer(),
) -> datasets.BaseDataset:
    """
    Load a dataset from dataset module.

    Parameters
    ----------
    name: str
        The name of the dataset to load.
    root: Optional[str]
        The path to the dataset. Default is None.
    
    Returns
    -------
    out: datasets.BaseDataset
        The dataset.
    """
    available_datasets = [
        "BaseDataset", "Biogen", "Efflux",
        "HClint", "HPPB", "RClint",
        "RPPB", "Solu", "DRD2",
        "FactorXA", "BACE",
        "BBBP", "ClinTox", "ESOL",
        "FreeSolv", "HIV", "Lipo",
        "MoleculeNet", "MUV", "MUV466",
        "MUV548", "MUV600", "MUV644", 
        "MUV652", "MUV689", "MUV692",
        "MUV712", "MUV713", "MUV733",
        "MUV737", "MUV810", "MUV832",
        "MUV846", "MUV852", "MUV858",
        "MUV859", "SIDER", "Tox21",
        "ToxCast", "QMugs"
    ]
    # assert name in available_datasets, f'Invalid dataset name. \
    #     Must be one of {available_datasets}.'
    print(f'Loading {name}...') if verbose else None
    return datasets.__dict__[name](
        root=root, compression=compression, verbose=verbose,
        standardizer=standardizer
    )


def load_molecules(
    dataset: datasets.BaseDataset|pd.DataFrame,
    verbose: bool = False
):
    """
    Load molecules from a dataset. Molecules are not standardized by this function.

    Parameters
    ----------
    dataset: datasets.BaseDataset
        The dataset to load molecules from. Must have a 'SMILES' column.

    Returns
    -------
    out: list[Chem.Mol]
        The molecules.
    """
    smiles = dataset['SMILES'].tolist()
    mols = [
        Chem.MolFromSmiles(i)
        for i in tqdm(smiles, desc='Loading molecules', disable=not verbose)
    ]
    return mols

def numpy_to_rdkit(array: np.ndarray):
    """
    Function to convert a numpy array to an RDKit explicit bit vector.

    Parameters
    ----------
    array: np.ndarray
        The binary array.
    
    Returns
    -------
    out: DataStructs.ExplicitBitVect
        The RDKit explicit bit vector.
    """
    assert array.ndim == 1, 'Array must be 1D.'
    assert ((array[0]==0) | (array[0]==1)).all(), 'Array must be binary.'
    out = DataStructs.ExplicitBitVect(len(array))
    indexes = np.where(array)[0].tolist()
    out.SetBitsFromList(indexes)
    return out