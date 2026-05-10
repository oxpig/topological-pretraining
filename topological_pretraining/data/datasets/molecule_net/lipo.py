from pathlib import Path

from ...mol import Standardizer
from ..base import BaseDataFrame


class Lipo(BaseDataFrame):
	url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv'

	def __init__(
		self,
		root: str | None = None,
		compression: bool = True,
		verbose: bool = True,
		standardizer: Standardizer = None,
	):
		suffix = 'csv.gz' if compression else 'csv'
		csv = Path(root) / f'Lipophilicity.{suffix}' if root else None
		super().__init__(
			csv=csv,
			url=self.url,
			compression=compression,
			verbose=verbose,
			standardizer=standardizer,
		)
		if 'SMILES' not in self.columns:
			self.rename(columns={'smiles': 'SMILES', 'exp': 'y'}, inplace=True)
			self.drop(self.columns.difference(['SMILES', 'y']), axis=1, inplace=True)
			self.mol_standardize_check()
			self.save()

	@property
	def task(self):
		return 'regression'
