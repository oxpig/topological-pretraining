from ..base import BaseDataset
from pathlib import Path

class Tox21(BaseDataset):
    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz'

    def __init__(self, root: str|None = None, compression: bool = True):
        suffix = 'csv.gz' if compression else 'csv'
        csv = Path(root) / f'tox21.{suffix}' if root else None
        super(Tox21, self).__init__(csv=csv, url=self.url, compression=compression)

        self.rename(
            columns={'smiles': 'SMILES'},
            inplace=True
        )
        self.save()

    @property
    def tasks(self):
        return 'binary classification'
    
    def save_subset(self, name: str, compression: bool = True):
        if self.csv is None:
            return None
        csv = Path(self.csv)
        suffixes = ''.join(csv.suffixes)
        csv = csv.parent / f'{name}{suffixes}'
        self.save(csv=csv, compression=compression)

class Tox21_Subset(Tox21, BaseDataset):

    def __init__(self, root: str|None = None, compression: bool = True):
        compression = '.gz' if compression else ''
        csv = Path(root) / f'tox21_{self.name.lower()}.csv{compression}' if root else None
        if csv is None or not csv.exists():
            Tox21.__init__(self, root=root, compression=compression)
            col = self.name.replace('_', '-')
            self.rename(
                columns={col: 'y'},
                inplace=True
            )
            self.drop(
                self.columns.difference(['SMILES', 'y']),
                axis=1, inplace=True
            )
            self.dropna(subset=['y'], inplace=True)
            self.save(csv)
        else:
            BaseDataset.__init__(self, csv=csv, compression=compression)

class NR_AR(Tox21_Subset):

    def __init__(self, root: str|None = None, compression: bool = True):
        super(NR_AR, self).__init__(root=root, compression=compression)

class NR_AR_LBD(Tox21_Subset):

    def __init__(self, root: str|None = None, compression: bool = True):
        super(NR_AR_LBD, self).__init__(root=root, compression=compression)

class NR_AhR(Tox21_Subset):

    def __init__(self, root: str|None = None, compression: bool = True):
        super(NR_AhR, self).__init__(root=root, compression=compression)

class NR_Aromatase(Tox21_Subset):
    def __init__(self, root: str|None = None, compression: bool = True):
        super(NR_Aromatase, self).__init__(root=root, compression=compression)
        
class NR_ER(Tox21_Subset):
    def __init__(self, root: str|None = None, compression: bool = True):
        super(NR_ER, self).__init__(root=root, compression=compression)

class NR_ER_LBD(Tox21_Subset):
    def __init__(self, root: str|None = None, compression: bool = True):
        super(NR_ER_LBD, self).__init__(root=root, compression=compression)

class NR_PPAR_gamma(Tox21_Subset):
    def __init__(self, root: str|None = None, compression: bool = True):
        super(NR_PPAR_gamma, self).__init__(root=root, compression=compression)

class SR_ARE(Tox21_Subset):
    def __init__(self, root: str|None = None, compression: bool = True):
        super(SR_ARE, self).__init__(root=root, compression=compression)

class SR_ATAD5(Tox21_Subset):
    def __init__(self, root: str|None = None, compression: bool = True):
        super(SR_ATAD5, self).__init__(root=root, compression=compression)

class SR_HSE(Tox21_Subset):
    def __init__(self, root: str|None = None, compression: bool = True):
        super(SR_HSE, self).__init__(root=root, compression=compression)


class SR_MMP(Tox21_Subset):
    def __init__(self, root: str|None = None, compression: bool = True):
        super(SR_MMP, self).__init__(root=root, compression=compression)

class SR_p53(Tox21_Subset):
    def __init__(self, root: str|None = None, compression: bool = True):
        super(SR_p53, self).__init__(root=root, compression=compression)
