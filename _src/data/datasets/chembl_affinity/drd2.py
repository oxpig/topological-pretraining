from ..base import BaseDataset

from pathlib import Path

class DRD2(BaseDataset):

    url = 'https://raw.githubusercontent.com/MarkusFerdinandDablander/QSAR-activity-cliff-experiments/refs/heads/main/data/chembl_dopamine_d2/molecule_data_clean.csv'

    def __init__(self, root: str|None = None, compression: bool = True):
        csv = Path(root) / 'drd2.csv' if root else None
        super(DRD2, self).__init__(csv=csv, url=self.url, compression=compression)
        self.rename(
            columns={
                'Ki [nM]': 'y'
            },
            inplace=True,
        )
        self.save()