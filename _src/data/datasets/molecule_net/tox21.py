from ..base import BaseDataset
from pathlib import Path

class Tox21(BaseDataset):
    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz'

    def __init__(self, root: str|None = None, compression: bool = True):
        suffix = 'csv' if compression else 'csv.gz'
        csv = Path(root) / f'tox21.{suffix}' if root else None
        super(Tox21, self).__init__(csv=csv, url=self.url, compression=compression)