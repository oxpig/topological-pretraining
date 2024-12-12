from .base import BaseDataset
import pandas as pd
import os
from typing import Literal

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

    def __init__(self, root: str|None = None):
        csv = os.path.join(root, 'biogen.csv') if root is not None else None
        
        super(Biogen, self).__init__(csv=csv, url=self.url)
        self.rename(columns=self.col_names, inplace=True)
        self.root = root

    def get_subsets(self):
        """
        Returns the subsets of the dataset.

        Yields:
        --------
        pd.DataFrame:
            The subsets of the dataset.
        """
        for task in self.col_names.values():
            yield self.subset(task)

    def subset(self, name: str):
        """
        Returns the subset of the dataset.

        Parameters:
        -----------
        name: str
            The name of the subset to return.
        
        Returns:
        --------
        pd.DataFrame
            The subset of the dataset.
        """
        data = self[self[name].notna()]
        data = data.reset_index()
        data = data.rename(columns={name: 'y', 'index': 'biogen_index'})
        
        return data[['biogen_index', 'SMILES','y']]

    @property
    def rat_ppb(self):
        return self.subset('rat_ppb')

    @property
    def human_ppb(self):
        return self.subset('human_ppb')

    @property
    def solu(self):
        return self.subset('solu')

    @property
    def human_clint(self):
        return self.subset('human_clint')

    @property
    def rat_clint(self):
        return self.subset('rat_clint')

    @property
    def efflux(self):
        return self.subset('efflux')

class BiogenSubset(pd.DataFrame):
    """
    Subset of the Biogen dataset.
    
    Parameters:
    -----------
    root: str|None
        Optional root directory where the dataset will be stored or retrieved from.
    task: Literal['human_clint', 'rat_clint', 'human_ppb', 'rat_ppb', 'solu', 'efflux']
        The name of the task. Default is 'human_clint'.
        """
    
    def __init__(
        self,
        root: str|None = None,
        task: Literal[
            'human_clint', 'rat_clint', 'human_ppb',
            'rat_ppb', 'solu', 'efflux'
        ] = 'human_clint'
    ):
        csv = os.path.join(root, f'{task}.csv') if root is not None else None
        
        if csv is None or not os.path.exists(csv):
            df = Biogen(root=root)
            data = df.subset(task)
            if csv is not None:
                data.to_csv(csv, index=False)
        else:
            data = pd.read_csv(csv)

        super(BiogenSubset, self).__init__(data=data)
        self.root = root
        self.task = task
        
    
    def save(self):
        """
        Saves the dataset to disk.
        """
        self.to_csv(os.path.join(self.root, f'{self.task}.csv'), index=False)


class HPPB(BiogenSubset):
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
    def __init__(self, root: str|None = None):
        super(HPPB, self).__init__(root=root, task='human_ppb')

class RPPB(BiogenSubset):
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
    def __init__(self, root: str|None = None):
        super(RPPB, self).__init__(root=root, task='rat_ppb')

class Solu(BiogenSubset):
    """
    Solubility subset of the Biogen dataset.
    
    Target: LOG SOLUBILITY PH 6.8 (ug/mL)
    
    Parameters:
    -----------
    root: str|None
        Optional root directory where the dataset will be stored or retrieved from.

    For inherited attributes and methods, see the pandas.DataFrame:
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
    
    """
    def __init__(self, root: str|None = None):
        super(Solu, self).__init__(root=root, task='solu')
    
class HClint(BiogenSubset):
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
    def __init__(self, root: str|None = None):
        super(HClint, self).__init__(root=root, task='human_clint')

class RClint(BiogenSubset):
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
    def __init__(self, root: str|None = None):
        super(RClint, self).__init__(root=root, task='rat_clint')
    