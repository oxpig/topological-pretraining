from ..base import BaseDataset
from ...mol import Standardizer

import numpy as np
from pathlib import Path

class FactorXA(BaseDataset):

    url = 'https://raw.githubusercontent.com/MarkusFerdinandDablander/QSAR-activity-cliff-experiments/refs/heads/main/data/chembl_factor_xa/molecule_data_clean.csv'

    def __init__(
        self, root: str|None = None, compression: bool = True,
        verbose: bool = False, standardizer: Standardizer = Standardizer()
    ):
        suffix = 'csv.gz' if compression else 'csv'
        csv = Path(root) / f'factor_xa.{suffix}' if root else None
        super(FactorXA, self).__init__(
            csv=csv, url=self.url, compression=compression,
            verbose=verbose, standardizer=standardizer
        )
        if 'y' not in self.columns:
            self.rename(
                columns={
                    'Ki [nM]': 'y'
                },
                inplace=True,
            )
            self.y = self.y.apply(lambda x: -np.log10(x))
            self.mol_standardize_check()
            self.save()
    
    @property
    def task(self):
        return 'regression'