from ..base import BaseDataset

class MUV(BaseDataset):
    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/muv.csv.gz'

    def __init__(self, csv: str|None = None):
        super(MUV, self).__init__(csv=csv, url=self.url)