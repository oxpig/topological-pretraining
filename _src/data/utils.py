from . import datasets
from .mol import Standardizer

import pandas as pd
from rdkit import Chem
from tqdm import tqdm
from typing import Literal

def load_dataset(
    name: Literal[
        'Biogen', 'Efflux', 'HClint',
        'HPPB', 'RClint', 'RPPB',
        'Solu', 'MoleculeNet', 'QMugs'
    ],
    root: str|None = None,
    compression: bool = True
) -> datasets.BaseDataset:
    """
    Load a dataset from dataset module.

    Parameters
    ----------
    name: Literal[
        'Biogen', 'Efflux', 'HClint',
        'HPPB', 'RClint', 'RPPB',
        'Solu', 'MoleculeNet', 'QMugs'
    ]
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