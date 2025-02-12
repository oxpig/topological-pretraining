import numpy as np
from rdkit import Chem
import torch
import torch_geometric as pyg
from tqdm import tqdm
from typing import Callable, Optional
from sklearn.preprocessing import MinMaxScaler

from copy import deepcopy

from ..data.mol import MolDesc
from ..data.feature_selection import CoCorr, SelectAll

class BaseTokenizer:

    _transform = None
    ready = False
    fixed_transform_kwargs = {}
    precomputed = False

    def __init__(
        self,
        transform_kwargs: dict = {},
        verbose: bool = False,
        **kwargs,
    ):
        self.verbose = verbose

        transform_kwargs.update(self.fixed_transform_kwargs)
        self.set_transform(kwargs=transform_kwargs)

    def __call__(self, X: Chem.Mol|list[Chem.Mol]) -> np.ndarray:
        if not self.ready:
            raise ValueError('Tokenizer must be fit before calling.')
        X = self.transform(X)
        return X

    @property
    def transform(self):
        return self._transform
    
    def set_transform(self, kwargs):
        self.transform_kwargs = kwargs
        self._transform = self._transform_base(**kwargs)
        

    def _transform_base(self, **kwargs):
        raise NotImplementedError

    def fit(self, mols: Chem.Mol, y: Optional[np.ndarray] = None) -> None:
        raise NotImplementedError
    
    @property
    def name(self):
        return self.__class__.__name__
    
    def to_dict(self):
        return {
            'name': self.name,
            'transform_kwargs': self.transform_kwargs,
        }
    
class BaseGraph:

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def make_graph(self, mol: Chem.Mol):
        raise NotImplementedError
    
    def __call__(self, X: Chem.Mol|list[Chem.Mol]):

        if isinstance(X, Chem.Mol):
            return self.make_graph(X)

        out = []
        pbar = tqdm(total=len(X), desc='Generating graphs', disable=not self.verbose)
        for idx, mol in enumerate(X):
            if not isinstance(mol, Chem.Mol):
                out.append(None)
            else:
                out.append(self.make_graph(mol))
            pbar.update(1)
        pbar.close()
        return out
