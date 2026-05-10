from pathlib import Path

import numpy as np
from rdkit import Chem

from ..mol import Standardizer
from .base import BaseDataFrame


class Biogen(BaseDataFrame):
	"""
	Dataset from the paper:
	    Fang, C. et al. 2023. Prospective Validation of Machine Learning Algorithms for
	    Absorption, Distribution, Metabolism, and Excretion  Prediction: An Industrial Perspective.
	    J. Chem. Inf. Model. 63, 3263–3274. https://doi.org/10.1021/acs.jcim.3c00160

	The dataset contains 3521 molecules with 6 tasks:

	- human_clint: LOG HLM_CLint (mL/min/kg)
	- solu: LOG SOLUBILITY PH 6.8 (ug/mL)
	- human_ppb: LOG PLASMA PROTEIN BINDING (HUMAN) (% unbound)
	- rat_ppb: LOG PLASMA PROTEIN BINDING (RAT) (% unbound)
	- rat_clint: LOG RLM_CLint (mL/min/kg)
	- efflux: LOG MDR1-MDCK ER (B-A/A-B)

	The dataset is available at:
	    https://raw.githubusercontent.com/molecularinformatics/Computational-ADME/main/ADME_public_set_3521.csv

	Class inherits from BaseDataFrame; which in inherits from pandas.DataFrame.
	For inherited attributes and methods, see the pandas.DataFrame:
	    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html


	Parameters:
	-----------
	root : str | None
	    Optional root directory where the dataset will be stored.
	compression : bool | None
	    Whether to compress DataFrame when saving.
	verbose : bool
	    Boolean for verbosity. Default is `True`.
	standardizer : topological_pretraining.data.mol.Standardizer
	    An object for standardizing dataset molecules.
	    See `topological_pretraining.data.mol.Standardizer` for default.

	Attributes:
	-----------
	url: str
	    The URL of the dataset.
	col_names: dict[str, str]
	    Dictionary for renaming columns.
	root: str|None
	    The root directory where the dataset is stored.
	csv: str|None
	    The path to the dataset.

	Methods:
	--------
	get_subsets()
	    Yields the subsets of the dataset.
	subset(name: str)
	    Returns a subset of the dataset.
	    The subset is a DataFrame with columns:
	        'biogen_index' - The index of the molecule in the dataset.
	        'SMILES' - The original SMILES representation of the molecule.
	        'y' - The target value for the task.

	Properties:
	-----------
	rat_ppb: pd.DataFrame
	    The subset of the dataset for the rat plasma protein binding task.
	human_ppb: pd.DataFrame
	    The subset of the dataset for the human plasma protein binding task.
	solu: pd.DataFrame
	    The subset of the dataset for the solubility task.
	human_clint: pd.DataFrame
	    The subset of the dataset for the human clearance task.
	rat_clint: pd.DataFrame
	    The subset of the dataset for the rat clearance task.
	efflux: pd.DataFrame
	    The subset of the dataset for the efflux task.
	"""

	url = 'https://raw.githubusercontent.com/molecularinformatics/Computational-ADME/main/ADME_public_set_3521.csv'

	col_names: dict[str, str] = {
		'LOG HLM_CLint (mL/min/kg)': 'human_clint',
		'LOG SOLUBILITY PH 6.8 (ug/mL)': 'solu',
		'LOG PLASMA PROTEIN BINDING (HUMAN) (% unbound)': 'human_ppb',
		'LOG PLASMA PROTEIN BINDING (RAT) (% unbound)': 'rat_ppb',
		'LOG RLM_CLint (mL/min/kg)': 'rat_clint',
		'LOG MDR1-MDCK ER (B-A/A-B)': 'efflux',
	}

	def __init__(
		self,
		root: str | None = None,
		compression: bool = True,
		verbose: bool = True,
		standardizer: Standardizer = None,
	):
		suffix = 'csv.gz' if compression else 'csv'
		csv = Path(root) / f'biogen.{suffix}' if root is not None else None
		changes = not csv.exists() if csv is not None else True
		super().__init__(
			csv=csv, url=self.url, verbose=verbose, standardizer=standardizer
		)
		if changes:
			self.rename(columns=self.col_names, inplace=True)
			self.root = root
			self.mol_standardize_check()
			self.save(csv)


class Biogen_Subset(Biogen, BaseDataFrame):
	"""
	Class for defining a subset within Biogen (e.g., Human PPB data).

	Parameters:
	-----------
	root : str | None
	    Optional root directory where the dataset will be stored.
	compression : bool | None
	    Whether to compress DataFrame when saving.
	verbose : bool
	    Boolean for verbosity. Default is `True`.
	standardizer : topological_pretraining.data.mol.Standardizer
	    An object for standardizing dataset molecules.
	    See `topological_pretraining.data.mol.Standardizer` for default.
	"""

	def __init__(
		self,
		root: str | None = None,
		compression: bool = True,
		verbose: bool = True,
		standardizer: Standardizer = None,
	):
		compression = '.gz' if compression else ''
		csv = (
			Path(root) / f'biogen_{self.name.lower()}.csv{compression}'
			if root
			else None
		)
		if csv is None or not csv.exists():
			print('Loading Biogen dataset...') if verbose else None
			Biogen.__init__(
				self,
				root=root,
				compression=compression,
				verbose=verbose,
				standardizer=standardizer,
			)
			print(f'Creating subset {self.name}...') if verbose else None
			col = self.name.lower()
			self.rename(columns={col: 'y'}, inplace=True)
			self.drop(
				self.columns.difference(['SMILES', 'y', 'rdkit_pass']),
				axis=1,
				inplace=True,
			)
			self.dropna(subset=['y'], inplace=True)
			self['original_index'] = self.index
			self.reset_index(drop=True, inplace=True)
			if csv is not None:
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
	def mols_path(self):
		"""
		Get the shared csv file path for all subsets.

		Returns:
		-------
		str
		"""
		if self.csv is not None:
			return Path(self.csv).parent / 'biogen_molecules.npz'
		return None

	@property
	def rdkit_mols(self):
		"""
		Get molecules only in subset.
		"""
		if 'original_index' in self.columns:
			# for when the subset is created from the full dataset
			index = self['original_index'].values
		else:
			# for before the subset is created from the full dataset, when the index is the same as the original dataset
			index = self.index.values
		if self._rdkit_mols is not None:
			return self._rdkit_mols[index]
		elif self.mols_path is not None and self.mols_path.exists():
			mols = np.load(file=self.mols_path, allow_pickle=True)
			mols = mols['arr_0']
			mols = mols[index]
			return mols
		elif 'SMILES' in self.columns:
			print(
				'Running standardization check of molecules'
			) if self.verbose else None

			mols = self.iloc[index]['SMILES'].values
			mols = [Chem.MolFromSmiles(m, sanitize=False) for m in mols]
			mols = self.standardizer(mols)
			if self.mols_path is not None:
				np.savez_compressed(self.mols_path, mols)
			else:
				self._rdkit_mols = np.array(mols, dtype=object)
			return mols
		else:
			raise ValueError(
				'SMILES column not found in the dataset and no saved molecules'
			)

	@property
	def task(self):
		return 'regression'

	def units(self, subset: str | None = None):
		"""
		Returns the units of the target values for a subset.

		Parameters:
		-----------
		subset: str|None
		    The name of the subset. Default is None.

		Returns:
		--------
		str
		    The units of the target values.
		"""
		return {
			'human_clint': 'log$_{10}$(mL/min/kg)',
			'rat_clint': 'log$_{10}$(mL/min/kg)',
			'human_ppb': 'log$_{10}$(% Unbound)',
			'rat_ppb': 'log$_{10}$(% Unbound)',
			'solu': 'log$_{10}$(ug/mL)',
			'efflux': 'log([B-A]/[A-B])',
			'biogen_subset': None,
		}[subset]

	@property
	def unit(self):
		return self.units(self.name.lower())


class Human_PPB(Biogen_Subset):
	"""
	Human plasma protein binding subset of the Biogen dataset.

	Target: LOG PLASMA PROTEIN BINDING (HUMAN) (% unbound)

	Parameters:
	-----------
	root: str|None
	    Optional root directory where the dataset will be stored or retrieved from.

	For inherited attributes and methods, see the pandas.DataFrame:
	    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
	"""

	def __init__(
		self,
		root: str | None = None,
		compression: bool = True,
		verbose: bool = True,
		standardizer: Standardizer = None,
	):
		super().__init__(
			root=root,
			compression=compression,
			verbose=verbose,
			standardizer=standardizer,
		)


class Rat_PPB(Biogen_Subset):
	"""
	Rat plasma protein binding subset of the Biogen dataset.

	Target: LOG PLASMA PROTEIN BINDING (RAT) (% unbound)

	Parameters:
	-----------
	root: str|None
	    Optional root directory where the dataset will be stored or retrieved from.

	For inherited attributes and methods, see the pandas.DataFrame:
	    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html

	"""

	def __init__(
		self,
		root: str | None = None,
		compression: bool = True,
		verbose: bool = True,
		standardizer: Standardizer = None,
	):
		super().__init__(
			root=root,
			compression=compression,
			verbose=verbose,
			standardizer=standardizer,
		)


class Solu(Biogen_Subset):
	"""
	Solubility subset of the Biogen dataset.

	Target: LOG SOLUBILITY PH 6.8 (ug/mL)

	Parameters:
	-----------
	root: str|None
	    Optional root directory where the dataset will be stored or retrieved from.

	For inherited attributes and methods, see the pandas.DataFrame:
	    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
	HP
	"""

	def __init__(
		self,
		root: str | None = None,
		compression: bool = True,
		verbose: bool = True,
		standardizer: Standardizer = None,
	):
		super().__init__(
			root=root,
			compression=compression,
			verbose=verbose,
			standardizer=standardizer,
		)


class Human_CLint(Biogen_Subset):
	"""
	Human clearance subset of the Biogen dataset.

	Target: LOG HLM_CLint (mL/min/kg)

	Parameters:
	-----------
	root: str|None
	    Optional root directory where the dataset will be stored or retrieved from.

	For inherited attributes and methods, see the pandas.DataFrame:
	    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
	"""

	def __init__(
		self,
		root: str | None = None,
		compression: bool = True,
		verbose: bool = True,
		standardizer: Standardizer = None,
	):
		super().__init__(
			root=root,
			compression=compression,
			verbose=verbose,
			standardizer=standardizer,
		)


class Rat_CLint(Biogen_Subset):
	"""
	Rat clearance subset of the Biogen dataset.

	Target: LOG RLM_CLint (mL/min/kg)

	Parameters:
	-----------
	root: str|None
	    Optional root directory where the dataset will be stored or retrieved from.

	For inherited attributes and methods, see the pandas.DataFrame:
	    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
	"""

	def __init__(
		self,
		root: str | None = None,
		compression: bool = True,
		verbose: bool = True,
		standardizer: Standardizer = None,
	):
		super().__init__(
			root=root,
			compression=compression,
			verbose=verbose,
			standardizer=standardizer,
		)


class Efflux(Biogen_Subset):
	"""
	Efflux ratio subset of the Biogen dataset.

	Target: LOG MDR1-MDCK ER (B-A/A-B)

	Parameters:
	-----------
	root: str|None
	    Optional root directory where the dataset will be stored or retrieved from.

	For inherited attributes and methods, see the pandas.DataFrame:
	    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
	"""

	def __init__(
		self,
		root: str | None = None,
		compression: bool = True,
		verbose: bool = True,
		standardizer: Standardizer = None,
	):
		super().__init__(
			root=root,
			compression=compression,
			verbose=verbose,
			standardizer=standardizer,
		)
