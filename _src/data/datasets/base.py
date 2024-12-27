import os
import pandas as pd
from pathlib import Path

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
    def __init__(
        self,
        data: pd.DataFrame|None = None,
        csv: str|None = None,
        url: str|None = None,
        compression: bool = True
    ):

        # Check if csv or url is provided
        if csv is None and url is None and data is None:
            raise ValueError('Either csv, url, or data must be provided')
        
        if data is not None:
            df = data
        
        elif csv is None or not os.path.exists(csv):
            # Download the csv file
            assert url is not None, 'URL must be provided if CSV does not exist'
            df = pd.read_csv(url)
        else:
            df = pd.read_csv(csv)

        if csv is not None and not os.path.exists(csv):
            # Save the csv file
            df.to_csv(
                csv, index=False, compression='infer' if compression else None
            )
        

        # Initialize the DataFrame
        super(BaseDataset, self).__init__(data=df)
        # Set the csv, url, and compression
        self.csv = csv
        self.url = url
        self.compression = compression

    @property
    def name(self):
        """
        Get the name of the dataset.
        """
        return self.__class__.__name__
    
    def save(self, csv: str|None = None, compression: bool|None = None):
        """
        Save the dataset to a csv file.
        """
        if compression is not None:
            self.compression = compression
        if csv is not None:
            self.csv = csv
        if self.csv is not None:
            self.to_csv(
                self.csv,
                index=False,
                compression='infer' if self.compression else None
            )

    @property
    def task(self):
        """
        Get the task of the dataset as string, e.g., 'regression'.
        """
        raise NotImplementedError