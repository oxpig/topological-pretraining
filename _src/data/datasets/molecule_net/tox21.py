from ..base import BaseDataset

class Tox21(BaseDataset):
    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz'

    def __init__(self, csv: str|None = None):
        super(Tox21, self).__init__(csv=csv, url=self.url)