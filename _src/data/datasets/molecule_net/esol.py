from ..base import BaseDataset
import numpy as np

class ESOL(BaseDataset):
    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv'

    def __init__(self, csv: str|None = None):
        super(ESOL, self).__init__(csv=csv, url=self.url)
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