from .base import BaseDataset
import pandas as pd
import os
from typing import Literal
from pathlib import Path

class Biogen(BaseDataset):
    """
    Dataset from the paper:
        Fang, C. et al. 2023. Prospective Validation of Machine Learning Algorithms for
        Absorption, Distribution, Metabolism, and Excretion  Prediction: An Industrial Perspective.
        J. Chem. Inf. Model. 63, 3263–3274. https://doi.org/10.1021/acs.jcim.3c00160

    The dataset contains 3521 molecules with 6 tasks:

    - human_clint: LOG HLM_CLint (mL/min/kg)
    - solu: LOG SOLUBILITY PH 6.8 (ug/mL)
    - human_ppb: LOG PLASMA PROTEIN BINDING (HUMAN) (% unbound)
    - rat_ppb: LOG PLASMA PROTEIN BINDING (RAT) (% unbound)
    - rat_clint: LOG RLM_CLint (mL/min/kg)
    - efflux: LOG MDR1-MDCK ER (B-A/A-B)

    The dataset is available at: 
        https://raw.githubusercontent.com/molecularinformatics/Computational-ADME/main/ADME_public_set_3521.csv

    Class inherits from BaseDataset; which in inherits from pandas.DataFrame.
    For inherited attributes and methods, see the pandas.DataFrame:
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html


    Parameters:
    -----------
    root: str|None
        Optional root directory where the dataset will be stored.

    Attributes:
    -----------
    url: str
        The URL of the dataset.
    col_names: dict[str, str]
        Dictionary for renaming columns.
    root: str|None
        The root directory where the dataset is stored.
    csv: str|None
        The path to the dataset.
    
    Methods:
    --------
    get_subsets()
        Yields the subsets of the dataset.
    subset(name: str)
        Returns a subset of the dataset.
        The subset is a DataFrame with columns:
            'biogen_index' - The index of the molecule in the dataset.
            'SMILES' - The original SMILES representation of the molecule.
            'y' - The target value for the task.

    Properties:
    -----------
    rat_ppb: pd.DataFrame
        The subset of the dataset for the rat plasma protein binding task.
    human_ppb: pd.DataFrame
        The subset of the dataset for the human plasma protein binding task.
    solu: pd.DataFrame
        The subset of the dataset for the solubility task.
    human_clint: pd.DataFrame
        The subset of the dataset for the human clearance task.
    rat_clint: pd.DataFrame
        The subset of the dataset for the rat clearance task.
    efflux: pd.DataFrame
        The subset of the dataset for the efflux task.
    """
    url = "https://raw.githubusercontent.com/molecularinformatics/Computational-ADME/main/ADME_public_set_3521.csv"

    col_names: dict[str, str] = {
            'LOG HLM_CLint (mL/min/kg)': 'human_clint',
            'LOG SOLUBILITY PH 6.8 (ug/mL)': 'solu',
            'LOG PLASMA PROTEIN BINDING (HUMAN) (% unbound)': 'human_ppb',
            'LOG PLASMA PROTEIN BINDING (RAT) (% unbound)': 'rat_ppb',
            'LOG RLM_CLint (mL/min/kg)': 'rat_clint',
            'LOG MDR1-MDCK ER (B-A/A-B)': 'efflux',
        }

    def __init__(self, root: str|None = None, compression: bool = True):
        suffix = 'csv.gz' if compression else 'csv'
        csv = os.path.join(root, f'biogen.{suffix}') if root is not None else None
        changes = os.path.exists(changes) if csv is not None else True
        super(Biogen, self).__init__(csv=csv, url=self.url)
        if changes:
            self.rename(columns=self.col_names, inplace=True)
            self.root = root
            self.mol_standardize_check()
            self.save(csv)
        
class Biogen_Subset(Biogen, BaseDataset):

    def __init__(self, root: str|None = None, compression: bool = True):
        compression = '.gz' if compression else ''
        csv = Path(root) / f'biogen_{self.name.lower()}.csv{compression}' if root else None
        if csv is None or not csv.exists():
            Biogen.__init__(self, root=root, compression=compression)
            col = self.name
            self.rename(
                columns={col: 'y'},
                inplace=True
            )
            self.drop(
                self.columns.difference(['SMILES', 'y', 'rdkit_pass']),
                axis=1, inplace=True
            )
            self.dropna(subset=['y'], inplace=True)
            self.save(csv)
        else:
            BaseDataset.__init__(self, csv=csv, compression=compression)

    @property
    def task(self):
        return 'regression'

    def units(self, subset: str|None = None):
        """
        Returns the units of the target values for a subset.

        Parameters:
        -----------
        subset: str|None
            The name of the subset. Default is None.
        
        Returns:
        --------
        str
            The units of the target values.
        """
        return {
            'human_clint': 'log$_{10}$(mL/min/kg)',
            'rat_clint': 'log$_{10}$(mL/min/kg)',
            'human_ppb': 'log$_{10}$(% Unbound)',
            'rat_ppb': 'log$_{10}$(% Unbound)',
            'solu': 'log$_{10}$(ug/mL)',
            'efflux': 'log([B-A]/[A-B])',
            'biogen_subset': None
        }[subset]

    @property
    def unit(self):
        return self.units(self.name.lower())



class Human_PPB(Biogen_Subset):
    """
    Human plasma protein binding subset of the Biogen dataset.

    Target: LOG PLASMA PROTEIN BINDING (HUMAN) (% unbound)

    Parameters:
    -----------
    root: str|None
        Optional root directory where the dataset will be stored or retrieved from.

    For inherited attributes and methods, see the pandas.DataFrame:
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
    """
    def __init__(self, root: str|None = None, compression: bool = True):
        super(Human_PPB, self).__init__(root=root, compression=compression)

class Rat_PPB(Biogen_Subset):
    """
    Rat plasma protein binding subset of the Biogen dataset.
    
    Target: LOG PLASMA PROTEIN BINDING (RAT) (% unbound)
    
    Parameters:
    -----------
    root: str|None
        Optional root directory where the dataset will be stored or retrieved from.

    For inherited attributes and methods, see the pandas.DataFrame:
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
    
    """
    def __init__(self, root: str|None = None, compression: bool = True):
        super(Rat_PPB, self).__init__(root=root, compression=compression)

class Solu(Biogen_Subset):
    """
    Solubility subset of the Biogen dataset.
    
    Target: LOG SOLUBILITY PH 6.8 (ug/mL)
    
    Parameters:
    -----------
    root: str|None
        Optional root directory where the dataset will be stored or retrieved from.

    For inherited attributes and methods, see the pandas.DataFrame:
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
    HP
    """
    def __init__(self, root: str|None = None, compression: bool = True):
        super(Solu, self).__init__(root=root, compression=compression)
    
class Human_CLint(Biogen_Subset):
    """
    Human clearance subset of the Biogen dataset.

    Target: LOG HLM_CLint (mL/min/kg)

    Parameters:
    -----------
    root: str|None
        Optional root directory where the dataset will be stored or retrieved from.

    For inherited attributes and methods, see the pandas.DataFrame:
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
    """
    def __init__(self, root: str|None = None, compression: bool = True):
        super(Human_CLint, self).__init__(root=root, compression=compression)

class Rat_CLint(Biogen_Subset):
    """
    Rat clearance subset of the Biogen dataset.

    Target: LOG RLM_CLint (mL/min/kg)

    Parameters:
    -----------
    root: str|None
        Optional root directory where the dataset will be stored or retrieved from.

    For inherited attributes and methods, see the pandas.DataFrame:
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
    """
    def __init__(self, root: str|None = None, compression: bool = True):
        super(Rat_CLint, self).__init__(root=root, compression=compression)
    
class Efflux(Biogen_Subset):
    """
    Efflux ratio subset of the Biogen dataset.

    Target: LOG MDR1-MDCK ER (B-A/A-B)

    Parameters:
    -----------
    root: str|None
        Optional root directory where the dataset will be stored or retrieved from.

    For inherited attributes and methods, see the pandas.DataFrame:
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
    """
    def __init__(self, root: str|None = None, compression: bool = True):
        super(Efflux, self).__init__(root=root, compression=compression)