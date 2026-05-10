from pathlib import Path

import numpy as np
from rdkit import Chem

from ...mol import Standardizer
from ..base import BaseDataFrame


class Tox21(BaseDataFrame):
	url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz'

	def __init__(
		self,
		root: str | None = None,
		compression: bool = True,
		verbose: bool = False,
		standardizer: Standardizer = None,
	):
		suffix = 'csv.gz' if compression else 'csv'
		csv = Path(root) / f'tox21.{suffix}' if root else None
		super().__init__(
			csv=csv,
			url=self.url,
			compression=compression,
			verbose=verbose,
			standardizer=standardizer,
		)
		if 'SMILES' not in self.columns:
			self.rename(columns={'smiles': 'SMILES'}, inplace=True)
			self.mol_standardize_check()
			self.save()

	@property
	def subsets(self):
		return [
			'NR_AR',
			'NR_AR_LBD',
			'NR_AhR',
			'NR_Aromatase',
			'NR_ER',
			'NR_ER_LBD',
			'NR_PPAR_gamma',
			'SR_ARE',
			'SR_ATAD5',
			'SR_HSE',
			'SR_MMP',
			'SR_p53',
		]


class Tox21_Subset(Tox21, BaseDataFrame):
	def __init__(
		self,
		root: str | None = None,
		compression: bool = True,
		verbose: bool = False,
		standardizer: Standardizer = None,
	):
		compression = '.gz' if compression else ''
		csv = (
			Path(root) / f'tox21_{self.name.lower()}.csv{compression}' if root else None
		)
		if csv is None or not csv.exists():
			Tox21.__init__(
				self,
				root=root,
				compression=compression,
				verbose=verbose,
				standardizer=standardizer,
			)

			col = self.name.replace('_', '-')
			self.rename(columns={col: 'y'}, inplace=True)
			self.drop(
				self.columns.difference(['SMILES', 'y', 'rdkit_pass']),
				axis=1,
				inplace=True,
			)
			self.dropna(subset=['y'], inplace=True)
			self['original_index'] = self.index
			self.reset_index(drop=True, inplace=True)
			self.save(csv)
		else:
			BaseDataFrame.__init__(
				self,
				csv=csv,
				compression=compression,
				verbose=verbose,
				standardizer=standardizer,
			)

	@property
	def task(self):
		return 'classification'

	@property
	def mols_path(self):
		if self.csv is not None:
			return Path(self.csv).parent / 'tox21_molecules.npz'
		return None

	@property
	def rdkit_mols(self):
		if self.mols_path is not None and self.mols_path.exists():
			mols = np.load(file=self.mols_path, allow_pickle=True)
			mols = mols['arr_0']
			mols = mols[self['original_index'].values]
			return mols

		elif 'SMILES' in self.columns:
			print(
				'Running standardization check of molecules'
			) if self.verbose else None
			mols = self['SMILES'].values
			mols = [Chem.MolFromSmiles(m, sanitize=False) for m in mols]
			mols = self.standardizer(mols)
			if self.mols_path is not None:
				np.savez_compressed(self.mols_path, mols)
			return mols
		else:
			raise ValueError(
				'SMILES column not found in the dataset and no saved molecules'
			)


"""
Target types:
    NR = Nuclear Receptor
    SR = Stress Response

Target abbreviations:
    AR = Androgen Receptor
    LBD = Ligand binding domain
    AhR = Aryl hydrocarbon receptor
    ER = Estrogen receptor
    PPAR = Peroxisome proliferator-activated receptor
    ARE = Antioxidant response element
    ATAD5 = ATPase family AAA domain containing 5
    HSE = Heat shock factor response element
    p53 = Tumor protein p53

Other abbreviations:
    MMP = Mitochondrial membrane potential
"""


class NR_AR(Tox21_Subset):
	def __init__(
		self,
		root: str | None = None,
		compression: bool = True,
		verbose: bool = False,
		standardizer: Standardizer = None,
	):
		super().__init__(
			root=root,
			compression=compression,
			verbose=verbose,
			standardizer=standardizer,
		)


class NR_AR_LBD(Tox21_Subset):
	def __init__(
		self,
		root: str | None = None,
		compression: bool = True,
		verbose: bool = False,
		standardizer: Standardizer = None,
	):
		super().__init__(
			root=root,
			compression=compression,
			verbose=verbose,
			standardizer=standardizer,
		)


class NR_AhR(Tox21_Subset):
	def __init__(
		self,
		root: str | None = None,
		compression: bool = True,
		verbose: bool = False,
		standardizer: Standardizer = None,
	):
		super().__init__(
			root=root,
			compression=compression,
			verbose=verbose,
			standardizer=standardizer,
		)


class NR_Aromatase(Tox21_Subset):
	def __init__(
		self,
		root: str | None = None,
		compression: bool = True,
		verbose: bool = False,
		standardizer: Standardizer = None,
	):
		super().__init__(
			root=root,
			compression=compression,
			verbose=verbose,
			standardizer=standardizer,
		)


class NR_ER(Tox21_Subset):
	def __init__(
		self,
		root: str | None = None,
		compression: bool = True,
		verbose: bool = False,
		standardizer: Standardizer = None,
	):
		super().__init__(
			root=root,
			compression=compression,
			verbose=verbose,
			standardizer=standardizer,
		)


class NR_ER_LBD(Tox21_Subset):
	def __init__(
		self,
		root: str | None = None,
		compression: bool = True,
		verbose: bool = False,
		standardizer: Standardizer = None,
	):
		super().__init__(
			root=root,
			compression=compression,
			verbose=verbose,
			standardizer=standardizer,
		)


class NR_PPAR_gamma(Tox21_Subset):
	def __init__(
		self,
		root: str | None = None,
		compression: bool = True,
		verbose: bool = False,
		standardizer: Standardizer = None,
	):
		super().__init__(
			root=root,
			compression=compression,
			verbose=verbose,
			standardizer=standardizer,
		)


class SR_ARE(Tox21_Subset):
	def __init__(
		self,
		root: str | None = None,
		compression: bool = True,
		verbose: bool = False,
		standardizer: Standardizer = None,
	):
		super().__init__(
			root=root,
			compression=compression,
			verbose=verbose,
			standardizer=standardizer,
		)


class SR_ATAD5(Tox21_Subset):
	def __init__(
		self,
		root: str | None = None,
		compression: bool = True,
		verbose: bool = False,
		standardizer: Standardizer = None,
	):
		super().__init__(
			root=root,
			compression=compression,
			verbose=verbose,
			standardizer=standardizer,
		)


class SR_HSE(Tox21_Subset):
	def __init__(
		self,
		root: str | None = None,
		compression: bool = True,
		verbose: bool = False,
		standardizer: Standardizer = None,
	):
		super().__init__(
			root=root,
			compression=compression,
			verbose=verbose,
			standardizer=standardizer,
		)


class SR_MMP(Tox21_Subset):
	def __init__(
		self,
		root: str | None = None,
		compression: bool = True,
		verbose: bool = False,
		standardizer: Standardizer = None,
	):
		super().__init__(
			root=root,
			compression=compression,
			verbose=verbose,
			standardizer=standardizer,
		)


class SR_p53(Tox21_Subset):
	def __init__(
		self,
		root: str | None = None,
		compression: bool = True,
		verbose: bool = False,
		standardizer: Standardizer = None,
	):
		super().__init__(
			root=root,
			compression=compression,
			verbose=verbose,
			standardizer=standardizer,
		)
