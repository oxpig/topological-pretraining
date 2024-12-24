from ..base import BaseDataset
from pathlib import Path

class ESOL(BaseDataset):
    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv'

    def __init__(self, root: str|None = None, compression: bool = True):
        suffix = 'csv' if compression else 'csv.gz'
        csv = Path(root) / f'esol.{suffix}' if root else None
        super(ESOL, self).__init__(csv=csv, url=self.url, compression=compression)
        self.rename(
            columns={
                'smiles': 'SMILES',
                'measured log solubility in mols per litre': 'y'
            },
            inplace=True,
        )
        # unit conversion from mol/L to uM
        self['y'] = self['y'].apply(lambda x: x + 6)
        self.drop(
            self.columns.difference(['SMILES', 'y']),
            axis=1, inplace=True
        )
        self.save()