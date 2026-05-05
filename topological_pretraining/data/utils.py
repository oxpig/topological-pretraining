from __future__ import annotations
from _src.data.datasets import __dict__ as dataset_classes
from _src.data.mol import Standardizer

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from tqdm import tqdm

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from _src.data.datasets import BaseDataFrame

def load_dataset(
    name: str,
    root: str|None = None,
    compression: bool = True,
    verbose: bool = False,
    standardizer: Standardizer = Standardizer(),
) -> BaseDataFrame:
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
    out: BaseDataFrame
        The dataset.
    """
    available_datasets = [
        "BaseDataFrame", "Biogen", "Efflux",
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
    print(f'Loading {name}...') if verbose else None
    return dataset_classes[name](
        root=root, compression=compression, verbose=verbose,
        standardizer=standardizer
    )


def load_molecules(
    dataset: BaseDataFrame|pd.DataFrame,
    verbose: bool = False
):
    """
    Load molecules from a dataset. Molecules are not standardized by this function.

    Parameters
    ----------
    dataset: datasets.BaseDataFrame
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
    if array.ndim != 1:
        raise ValueError('Array must be 1D.')
    if not ((array == 0) | (array == 1)).all():
        raise ValueError('Array must be binary.')
    out = DataStructs.ExplicitBitVect(len(array))
    indexes = np.where(array)[0].tolist()
    out.SetBitsFromList(indexes)
    return out