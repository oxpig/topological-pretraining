from ..base import BaseDataset

from pathlib import Path


class Lipo(BaseDataset):
    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv'

    def __init__(self, root: str|None = None, compression: bool = True):
        suffix = 'csv.gz' if compression else 'csv'
        csv = Path(root) / f'Lipophilicity.{suffix}' if root else None
        super(Lipo, self).__init__(csv=csv, url=self.url, compression=compression)

        self.rename(columns={'smiles': 'SMILES', 'exp': 'y'}, inplace=True)
        self.drop(
            self.columns.difference(['SMILES', 'y']),
            axis=1, inplace=True
        )
        self.save()

    @property
    def tasks(self):
        return 'regression'