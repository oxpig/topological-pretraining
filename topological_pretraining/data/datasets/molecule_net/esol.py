from pathlib import Path

from ...mol import Standardizer
from ..base import BaseDataFrame


class ESOL(BaseDataFrame):
	url = (
		'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv'
	)

	def __init__(
		self,
		root: str | None = None,
		compression: bool = True,
		verbose: bool = True,
		standardizer: Standardizer = None,
	):
		suffix = 'csv.gz' if compression else 'csv'
		csv = Path(root) / f'esol.{suffix}' if root else None
		super().__init__(
			csv=csv,
			url=self.url,
			compression=compression,
			verbose=verbose,
			standardizer=standardizer,
		)
		if 'SMILES' not in self.columns:
			self.rename(
				columns={
					'smiles': 'SMILES',
					'measured log solubility in mols per litre': 'y',
				},
				inplace=True,
			)
			# unit conversion from mol/L to uM
			self['y'] = self['y'].apply(lambda x: x + 6)
			self.drop(self.columns.difference(['SMILES', 'y']), axis=1, inplace=True)
			self.mol_standardize_check()
			self.save()

	@property
	def task(self):
		return 'regression'
