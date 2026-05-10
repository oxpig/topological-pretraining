from pathlib import Path

from ...mol import Standardizer
from ..base import BaseDataFrame


class MUV(BaseDataFrame):
	"""
	Maximum Unbiased Validation dataset.

	The Maximum Unbiased Validation (MUV) dataset is a collection of 17 assay results from
	PCBA, with filters for noisy/incorrect data. Each task is a binary classificaiton of
	activity against a particular target.

	This is better than the PCBA dataset as filters for erroneous data are applied and assays
	are separated into different tasks rather than aggrgated. In DeepChem, the 17 tasks are
	aggregated into a single dataset, with NaN values being replaced with 0. This should be
	avoided, as each SMILES with a NaN value has not been assessed on that target; presuming
	inactivity without a known experimental result will introduce noise.
	"""

	url = (
		'https://github.com/deepchem/deepchem/raw/refs/heads/master/datasets/muv.csv.gz'
	)

	def __init__(
		self,
		root: str | None = None,
		compression: bool = True,
		verbose: bool = True,
		standardizer: Standardizer = None,
	):
		suffix = 'csv.gz' if compression else 'csv'
		csv = Path(root) / f'muv.{suffix}' if root else None
		super().__init__(
			csv=csv,
			url=self.url,
			compression=compression,
			verbose=verbose,
			standardizer=standardizer,
		)
		if 'SMILES' not in self.columns:
			self.rename(columns={'smiles': 'SMILES'}, inplace=True)
			if root is not None:
				self.mol_standardize_check()
			self.save()

	@property
	def task(self):
		return 'classification'

	@property
	def subsets(self):
		return [
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
		]


class MUV_Subset(MUV, BaseDataFrame):
	aid_number = None
	# for holding mols in memory if no path provided
	_rdkit_mols = None

	def __init__(
		self,
		root: str | None = None,
		compression: bool = True,
		verbose: bool = True,
		standardizer: Standardizer = None,
	):
		compression = '.gz' if compression else ''
		csv = Path(root) / f'muv{self.aid_number}.csv{compression}' if root else None
		mols_path = Path(root) / 'muv_molecules.npz' if root else None
		if csv is None or not csv.exists():
			MUV.__init__(
				self,
				root=root,
				compression=compression,
				verbose=verbose,
				standardizer=standardizer,
			)
			self.rename(columns={f'MUV-{self.aid_number}': 'y'}, inplace=True)
			self.drop(
				self.columns.difference(['SMILES', 'y', 'rdkit_pass']),
				axis=1,
				inplace=True,
			)
			self.dropna(subset=['y'], inplace=True)
			self['original_index'] = self.index
			self.reset_index(drop=True, inplace=True)
			if root is None:
				self.mol_standardize_check()
			self.save(csv)
		elif csv.exists() and not mols_path.exists():
			MUV.__init__(
				self,
				root=root,
				compression=compression,
				verbose=verbose,
				standardizer=standardizer,
			)
			BaseDataFrame.__init__(
				self,
				csv=csv,
				url=self.url,
				compression=compression,
				verbose=verbose,
				standardizer=standardizer,
			)
		else:
			BaseDataFrame.__init__(
				self,
				csv=csv,
				url=self.url,
				compression=compression,
				verbose=verbose,
				standardizer=standardizer,
			)

	@property
	def mols_path(self):
		if self.csv is not None:
			return Path(self.csv).parent / 'muv_molecules.npz'
		return None

	@property
	def rdkit_mols(self):
		mols = BaseDataFrame.rdkit_mols.fget(self)
		# if subset has been preprocessed with original_index column, select subset mols
		if (
			self.mols_path is not None
			and self.mols_path.exists()
			and 'original_index' in self.columns
		):
			mols = mols[self['original_index'].values]
		return mols

	@property
	def task(self):
		return 'classification'


class MUV466(MUV_Subset):
	"""
	MUV dataset for PCBA assay AID 466; S1P1 Agonists.

	https://pubchem.ncbi.nlm.nih.gov/bioassay/466

	This is a subset of the MUV dataset, with only the S1P1 Agonists assay.
	"""

	aid_number = 466

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


class MUV548(MUV_Subset):
	"""
	MUV dataset for PCBA assay AID 548; protein kinase A (PKA) inhibitors.

	https://pubchem.ncbi.nlm.nih.gov/bioassay/548

	This is a subset of the MUV dataset, with only the PKA inhibitors assay.
	"""

	aid_number = 548

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


class MUV600(MUV_Subset):
	"""
	MUV dataset for PCBA assay AID 600; nuclear receptor Steroidogenic Factor 1 (SF-1) inhibitors.

	https://pubchem.ncbi.nlm.nih.gov/bioassay/600

	This is a subset of the MUV dataset, with only the SF-1 inhibitors assay.
	"""

	aid_number = 600

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


class MUV644(MUV_Subset):
	"""
	MUV dataset for PCBA assay AID 644; Rho kinase 2 (Rock2) inhibitors.

	https://pubchem.ncbi.nlm.nih.gov/bioassay/644

	This is a subset of the MUV dataset, with only the Rock2 inhibitors assay.
	"""

	aid_number = 644

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


class MUV652(MUV_Subset):
	"""
	MUV dataset for PCBA assay AID 652; HIV-1 RT-RNase H inhibitors.

	https://pubchem.ncbi.nlm.nih.gov/bioassay/652

	This is a subset of the MUV dataset, with only the RT-RNH inhibitors assay.
	"""

	aid_number = 652

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


class MUV689(MUV_Subset):
	"""
	MUV dataset for PCBA assay AID 689; ephrin type-A receptor 4 precursor antagonists.

	https://pubchem.ncbi.nlm.nih.gov/bioassay/689

	This is a subset of the MUV dataset, with only the EphA4 receptor antagonists assay.
	"""

	aid_number = 689

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


class MUV692(MUV_Subset):
	"""
	MUV dataset for PCBA assay AID 692; nuclear receptor Steroidogenic Factor 1 (SF-1) activators.

	https://pubchem.ncbi.nlm.nih.gov/bioassay/692

	This is a subset of the MUV dataset, with only the SF-1 activators assay.
	"""

	aid_number = 692

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


class MUV712(MUV_Subset):
	"""
	MUV dataset for PCBA assay AID 712; tumor Hsp90 inhibitors.

	https://pubchem.ncbi.nlm.nih.gov/bioassay/712

	This is a subset of the MUV dataset, with only the Hsp90 inhibitors assay.
	"""

	aid_number = 712

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


class MUV713(MUV_Subset):
	"""
	MUV dataset for PCBA assay AID 713; Estrogen Receptor-alpha Coactivator Binding Inhibitors

	https://pubchem.ncbi.nlm.nih.gov/bioassay/713

	This is a subset of the MUV dataset, with only the ER-alpha inhibitors assay.
	"""

	aid_number = 713

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


class MUV733(MUV_Subset):
	"""
	MUV dataset for PCBA assay AID 733; Estrogen Receptor-beta Coactivator Binding Inhibitors

	https://pubchem.ncbi.nlm.nih.gov/bioassay/733

	This is a subset of the MUV dataset, with only the ER-beta inhibitors assay.
	"""

	aid_number = 733

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


class MUV737(MUV_Subset):
	"""
	MUV dataset for PCBA assay AID 737; Estrogen Receptor-alpha Coactivator Binding Potentiators.

	https://pubchem.ncbi.nlm.nih.gov/bioassay/737

	This is a subset of the MUV dataset, with only the ER-alpha potentiators assay.
	"""

	aid_number = 737

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


class MUV810(MUV_Subset):
	"""
	MUV dataset for PCBA assay AID 810; Focal Adhesion Kinase (FAK) inhibitors.

	https://pubchem.ncbi.nlm.nih.gov/bioassay/810

	This is a subset of the MUV dataset, with only the FAK inhibitors assay.
	"""

	aid_number = 810

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


class MUV832(MUV_Subset):
	"""
	MUV dataset for PCBA assay AID 832; Cathepsin G inhibitors.

	https://pubchem.ncbi.nlm.nih.gov/bioassay/832

	This is a subset of the MUV dataset, with only the Cathepsin G inhibitors assay.
	"""

	aid_number = 832

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


class MUV846(MUV_Subset):
	"""
	MUV dataset for PCBA assay AID 846; Factor XIa inhibitors.

	https://pubchem.ncbi.nlm.nih.gov/bioassay/846

	This is a subset of the MUV dataset, with only the Factor XIa inhibitors assay.
	"""

	aid_number = 846

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


class MUV852(MUV_Subset):
	"""
	MUV dataset for PCBA assay AID 852; Factor XIIa inhibitors.

	https://pubchem.ncbi.nlm.nih.gov/bioassay/852

	This is a subset of the MUV dataset, with only the Factor XIIa inhibitors assay.
	"""

	aid_number = 852

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


class MUV858(MUV_Subset):
	"""
	MUV dataset for PCBA assay AID 858; DRD1 allosteric modulators.

	https://pubchem.ncbi.nlm.nih.gov/bioassay/858

	This is a subset of the MUV dataset, with only the DRD1 allosteric modulators assay.
	"""

	aid_number = 858

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


class MUV859(MUV_Subset):
	"""
	MUV dataset for PCBA assay AID 859; M1 muscarinic acetylcholine receptor antagonists.

	https://pubchem.ncbi.nlm.nih.gov/bioassay/859

	This is a subset of the MUV dataset, with only the mAChR antagonists assay.
	"""

	aid_number = 859

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
