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
    compression: bool = True
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
    return datasets.__dict__[name](root=root, compression=compression)


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