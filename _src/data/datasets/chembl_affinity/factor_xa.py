from ..base import BaseDataset

from pathlib import Path

class FactorXA(BaseDataset):

    url = 'https://raw.githubusercontent.com/MarkusFerdinandDablander/QSAR-activity-cliff-experiments/refs/heads/main/data/chembl_factor_xa/molecule_data_clean.csv'

    def __init__(self, root: str|None = None, compression: bool = True):
        csv = Path(root) / 'factor_xa.csv' if root else None
        super(FactorXA, self).__init__(csv=csv, url=self.url, compression=compression)
        if 'y' not in self.columns:
            self.rename(
                columns={
                    'Ki [nM]': 'y'
                },
                inplace=True,
            )
            self.mol_standardize_check()
            self.save()