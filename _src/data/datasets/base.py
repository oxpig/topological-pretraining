import os
import pandas as pd
from pathlib import Path

from ..mol import Sanitzer

class BaseDataset(pd.DataFrame):

    def __init__(self, csv: str|None = None, url: str|None = None):

        if csv is None and url is None:
            raise ValueError('Either path or url must be provided')
        
        if csv is None or not os.path.exists(csv):
            assert url is not None, 'URL must be provided if CSV does not exist'
            df = pd.read_csv(url)
            if csv is not None:
                df.to_csv(csv, index=False)
        else:
            df = pd.read_csv(csv)

        super(BaseDataset, self).__init__(data=df)
        self.csv = csv
        self.url = url

    def sanitize(self):
        """
        TODO: Check SMILES validity and canonicalize them.
        """
        pass