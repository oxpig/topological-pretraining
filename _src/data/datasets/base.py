import os
import pandas as pd
from pathlib import Path

from ..mol import Sanitzer

class BaseDataset(pd.DataFrame):
    """
    Base pandas DataFrame class for datasets.

    Parameters
    ----------
    csv: str
        The path to the csv file.
    url: str
        The URL to download the csv file.
        URL is ignored if csv is provided and exists.
    compression: bool
        Whether to compress the dataset or not.
        Default is True.

    Attributes
    ----------
    csv: str
        The path to the csv file.
    url: str
        The URL to download the csv file.
    compression: bool
        Whether the saved csv is compressed or not.
    """

    def __init__(self, csv: str|None = None, url: str|None = None, compression: bool = True):

        # Check if csv or url is provided
        if csv is None and url is None:
            raise ValueError('Either path or url must be provided')
        
        if csv is None or not os.path.exists(csv):
            # Download the csv file
            assert url is not None, 'URL must be provided if CSV does not exist'
            df = pd.read_csv(url)
            if csv is not None:
                # Save the csv file
                df.to_csv(csv, index=False, compression=compression)
        else:
            # Load the csv filex
            df = pd.read_csv(csv)

        # Initialize the DataFrame
        super(BaseDataset, self).__init__(data=df)
        # Set the csv, url, and compression
        self.csv = csv
        self.url = url
        self.compression = compression


    def sanitize(self):
        """
        TODO: Check SMILES validity and canonicalize them.
        """
        pass