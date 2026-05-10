from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from tqdm import tqdm

from .datasets import __dict__ as dataset_classes
from .mol import Standardizer

if TYPE_CHECKING:
	from .datasets import BaseDataFrame

available_datasets: list[str] = [
	'Biogen',
	'Efflux',
	'Human_CLint',
	'Human_PPB',
	'Rat_CLint',
	'Rat_PPB',
	'Solu',
	'DRD2',
	'FactorXA',
	'ESOL',
	'FreeSolv',
	'Lipo',
	'MoleculeNet',
	'MUV',
	'MUV466',
	'MUV548',
	'MUV600',
	'MUV644',
	'MUV652',
	'MUV689',
	'MUV692',
	'MUV712',
	'MUV713',
	'MUV733',
	'MUV737',
	'MUV810',
	'MUV832',
	'MUV846',
	'MUV852',
	'MUV858',
	'MUV859',
	'Tox21',
	'NR_AR',
	'NR_AR_LBD',
	'NR_ER',
	'NR_ER_LBD',
	'SR_ARE',
	'SR_ATAD5',
	'SR_HSE',
	'SR_MMP',
	'NR_AhR',
	'NR_Aromatase',
	'NR_PPAR_gamma',
	'SR_p53',
	'Tox21',
	'QMugs',
]


def load_dataset(
	name: str,
	root: str | None = None,
	compression: bool = True,
	verbose: bool = False,
	standardizer: Standardizer | dict = None,
) -> BaseDataFrame:
	"""
	Load a dataset from dataset module.

	Parameters
	----------
	name: str
	    The name of the dataset to load.
	root: Optional[str]
	    The path to the dataset. Default is None.
	compression: bool
	    Whether to save the dataset with compression. Default is True.
	verbose: bool
	    Whether to print verbose output. Default is False.
	standardizer: Standardizer | dict
	    The standardizer to use. Default is a new Standardizer instance.

	Returns
	-------
	out: BaseDataFrame
	    The dataset.
	"""
	standardizer = standardizer or Standardizer()
	print(f'Loading {name}...') if verbose else None
	if name not in available_datasets:
		raise ValueError(
			f'Dataset {name} not found. Available datasets: {available_datasets}'
		)
	return dataset_classes[name](
		root=root, compression=compression, verbose=verbose, standardizer=standardizer
	)


def load_molecules(dataset: BaseDataFrame | pd.DataFrame, verbose: bool = False):
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
