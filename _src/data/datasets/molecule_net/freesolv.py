from ..base import BaseDataset

class FreeSolv(BaseDataset):
    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv'

    def __init__(self, csv: str|None = None):
        super(FreeSolv, self).__init__(csv=csv, url=self.url)
