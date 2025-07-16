import numpy as np
import os
import pandas as pd
from pathlib import Path
from rdkit import Chem
from typing import Callable

from ..mol import Standardizer

class BaseDataFrame(pd.DataFrame):
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
        compression: bool = True,
        verbose: bool = False,
        standardizer: Standardizer = Standardizer(),
    ):
        standardizer.verbose = verbose
        # Check if csv or url is provided
        if csv is None and url is None and data is None:
            raise ValueError('Either csv, url, or data must be provided')
        
        if data is not None:
            print('Setting input data as DataFrame.') if verbose else None
            df = data
        
        elif csv is None or not os.path.exists(csv):
            # Download the csv file
            assert url is not None, 'URL must be provided if CSV does not exist'
            print(f'Downloading csv from url...') if verbose else None
            df = pd.read_csv(url)
        else:
            print(f'Reading csv from path...') if verbose else None
            df = pd.read_csv(csv)

        if csv is not None and not os.path.exists(csv):
            # Save the csv file
            print(f'Saving csv to path...') if verbose else None
            df.to_csv(
                csv, index=False, compression='infer' if compression else None
            )
        

        # Initialize the DataFrame
        super(BaseDataFrame, self).__init__(data=df)
        # Set the csv, url, and compression
        self.csv = csv
        self.url = url
        self.compression = compression
        self.verbose = verbose
        self.standardizer = standardizer

    @property
    def name(self):
        """
        Returns:
        -------
        str
            Dataset name
        """
        return self.__class__.__name__
    
    def save(self, csv: str|None = None, compression: bool|None = None):
        """
        Save the dataset to a csv file.

        Parameters:
        ----------
        csv : str | None
            Path to save dataframe to csv. If `None` dataframe is not saved.
        compression : bool | None
            Whether to infer compression from the csv path.
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
    
    @property
    def rdkit_mols(self):
        """
        Get dataset molecules.
        Loaded from disk if a path exists, otherwise DataFrame SMILES are converted into RDKit 
        molecules and standardardized.

        By default, sanitization does not occur unless specified in the Standardizer.

        Returns:
        -------
        list[rdkit.Chem.Mol | None]
            A list of rdkit molecule objects.
        
        Raises:
        ------
        ValueError
            If no saved molecules or SMILES column in the dataset.
        """
        if self.mols_path is not None and self.mols_path.exists():
            mols = np.load(file=self.mols_path, allow_pickle=True)
            mols = mols['arr_0']
            return mols

        elif 'SMILES' in self.columns:
            print('Running standardization check of molecules') if self.verbose else None
            mols = self['SMILES'].values
            mols = [Chem.MolFromSmiles(m, sanitize=False) for m in mols]
            mols = self.standardizer(mols)
            
            if self.mols_path is not None:
                np.savez_compressed(self.mols_path, mols)
            return mols
        else:
            raise ValueError('SMILES column not found in the dataset and no saved molecules')
    
    def mol_standardize_check(self):
        """
        Adds `rdkit_pass` column of booleans to the DataFrame to indicate whether SMILES pass
        standardization.
        """
        if 'rdkit_pass' in self.columns:
            return
        mols = self.rdkit_mols
        out = np.where(np.array(mols) == None, False, True)
        self['rdkit_pass'] = out
        self.save()

    @property
    def mols_path(self):
        """
        Path to save RDKit molecules to.
        Saves as csv path with suffix changed from `.csv` to `.npz`

        Returns:
        -------
        str | None
            Return a path to save molecules to.
            Returns `None` if no csv path exists.
        """
        if self.csv is None:
            return None
        else:
            mols_path = Path(self.csv)
            mols_path = mols_path.parent / f'{mols_path.stem.split(".")[0]}.npz'
            return mols_path

    @property
    def splits(self):
        """
        Generator for yielding pre-defined train-test splits.
        Split columns must have a heading containing `split`,
        and contain `Train` and `Test` strings.

        Yields:
        ------
        numpy.ndarray
            Train indexes.
        numpy.ndarray
            Test indexes.
        """
        columns = [col for col in self.columns if 'split' in col]
        columns = list(sorted(columns, key=lambda x: int(x.split('_')[1])))
        for col in columns:
            if 'split' in col:
                col = self.loc[:, col]
                train = col[col == 'Train'].index.to_numpy()
                test = col[col == 'Test'].index.to_numpy()
                yield train, test

    @property
    def num_splits(self):
        """
        Count the number of splits in the DataFrame.

        Returns:
        -------
        int
            The total number of columns containing `split` in their name.
        """
        return len([col for col in self.columns if 'split' in col])

    @property
    def hyperopt_average(self):
        return 'mean'
    
    @property
    def splits_to_exclude_for_metrics(self):
        split_cols = [col for col in self.columns if 'split' in col]
        exclude = []
        for col in split_cols:
            if self.y[self[col] == "Test"].nunique() == 1:
                exclude.append(int(col.split('_')[1]))

        return np.array(exclude)