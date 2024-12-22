from ..base import BaseDataset

class FactorXA(BaseDataset):

    url = 'https://raw.githubusercontent.com/MarkusFerdinandDablander/QSAR-activity-cliff-experiments/refs/heads/main/data/chembl_factor_xa/molecule_data_clean.csv'

    def __init__(self, csv: str|None = None, compression: bool = True):
        super(FactorXA, self).__init__(csv=csv, url=self.url, compression=compression)