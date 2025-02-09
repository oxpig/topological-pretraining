from ..base import BaseDataset
from ...mol import Standardizer
from pathlib import Path

class FreeSolv(BaseDataset):
    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv'

    def __init__(
        self, root: str|None = None, compression: bool = True,
        verbose: bool = True, standardizer: Standardizer = Standardizer(),
    ):
        suffix = 'csv.gz' if compression else 'csv'
        csv = Path(root) / f'freesolv.{suffix}' if root else None
        super(FreeSolv, self).__init__(
            csv=csv, url=self.url, compression=compression, verbose=verbose, standardizer=standardizer
        )
        if 'SMILES' not in self.columns:
            self.rename(
                columns={'smiles': 'SMILES', 'expt': 'y'},
                inplace=True
            )
            self.drop(
                self.columns.difference(['SMILES', 'y', 'calc']),
                axis=1, inplace=True
            )
            self.mol_standardize_check()
            self.save()

    @property
    def task(self):
        return 'regression'