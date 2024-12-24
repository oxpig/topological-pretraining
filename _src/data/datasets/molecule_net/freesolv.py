from ..base import BaseDataset
from pathlib import Path

class FreeSolv(BaseDataset):
    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv'

    def __init__(self, root: str|None = None, compression: bool = True):
        suffix = 'csv' if compression else 'csv.gz'
        csv = Path(root) / f'freesolv.{suffix}' if root else None
        super(FreeSolv, self).__init__(csv=csv, url=self.url, compression=compression)
        self.rename(
            columns={'smiles': 'SMILES', 'expt': 'y'},
            inplace=True
        )
        self.drop(
            self.columns.difference(['SMILES', 'y', 'calc']),
            axis=1, inplace=True
        )
        self.save()