from ..base import BaseDataFrame
from ...mol import Standardizer

from pathlib import Path
import numpy as np

class DRD2(BaseDataFrame):

    url = 'https://raw.githubusercontent.com/MarkusFerdinandDablander/QSAR-activity-cliff-experiments/refs/heads/main/data/chembl_dopamine_d2/molecule_data_clean.csv'

    def __init__(
        self, root: str|None = None, compression: bool = True,
        verbose: bool = False, standardizer: Standardizer = Standardizer()
    ):
        suffix = 'csv.gz' if compression else 'csv'
        csv = Path(root) / f'drd2.{suffix}' if root else None
        super(DRD2, self).__init__(
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