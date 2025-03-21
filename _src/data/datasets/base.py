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
    
    @property
    def rdkit_mols(self):
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
        Standardize the dataset.
        """
        if 'rdkit_pass' in self.columns:
            return
        mols = self.rdkit_mols
        out = np.where(np.array(mols) == None, False, True)
        self['rdkit_pass'] = out
        self.save()

    @property
    def mols_path(self):
        if self.csv is None:
            return None
        else:
            mols_path = Path(self.csv)
            mols_path = mols_path.parent / f'{mols_path.stem.split(".")[0]}.npz'
            return mols_path

    @property
    def splits(self):
        for col in self.columns:
            if 'split' in col:
                col = self.loc[:, col]
                train = col[col == 'Train'].index.to_numpy()
                test = col[col == 'Test'].index.to_numpy()
                yield train, test

    @property
    def num_splits(self):
        return len([col for col in self.columns if 'split' in col])
    
    def save_standard_smiles(self):
        smi = [Chem.MolToSmiles(m) if m != None else None for m in self.rdkit_mols]
        try:
            mols_path = self.mols_path
            smi_path = mols_path.parent / f'{mols_path.stem}.smi'

            with open(smi_path, 'w') as f:
                for s in smi:
                    f.write(f'{s}\n')
        except:
            print('Could not save SMILES file')

    def get_smiles(self):
        mols_path = self.mols_path
        smi_path = mols_path.parent / f'{mols_path.stem}.smi'
        try:
            return pd.read_csv(smi_path, header=None).iloc[:,0].tolist()
        except:
            print('Could not load SMILES file')

    @property
    def hyperopt_average(self):
        return 'mean'