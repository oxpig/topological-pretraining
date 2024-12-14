from ..base import BaseDataset

class ESOL(BaseDataset):
    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv'

    def __init__(self, csv: str|None = None):
        super(ESOL, self).__init__(csv=csv, url=self.url)