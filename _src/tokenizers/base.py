import numpy as np
from rdkit import Chem
from tqdm import tqdm
from typing import Callable

from ..data.mol import MolDesc

class BaseTokenizer:

    _transform = None

    def __init__(
        self,
        X: list[Chem.Mol],
        y: np.ndarray = None,
        train: np.ndarray = np.array([]),
        test: np.ndarray = np.array([]),
        transform_kwargs: dict = {},
        verbose: bool = False,
    ):
        if len(train) == 0:
            train = np.arange(len(X))
        self.origin_X = X
        self.verbose = verbose
        self.set_transform(kwargs=transform_kwargs)
        self.X = self.transform(X)
        self.y = y
        self.train_idx = train
        self.test_idx = test
        self.transform_kwargs = transform_kwargs

    @property
    def transform(self):
        return self._transform
    
    def set_transform(self, kwargs):
        self._transform = self._transform_base(**kwargs)

    def _transform_base(self, **kwargs):
        raise NotImplementedError

    def reset(self, train: np.ndarray, test: np.ndarray) -> None:
        self.train_idx = train
        self.test_idx = test

    @property
    def train(self):
        if self.y is None:
            return self.X[self.train_idx]
        else:
            return self.X[self.train_idx], self.y[self.train_idx]
    
    @property
    def test(self):
        if self.y is None:
            return self.X[self.test_idx]
        else:
            return self.X[self.test_idx], self.y[self.test_idx]
    

class TestTokenizer(BaseTokenizer):

    def __init__(
        self,
        X,
        y = None,
        train = np.array([]),
        test = np.array([]), 
        transform_kwargs = {'descriptors': ['BalabanJ']}
    ):
        super().__init__(X, y, train, test, transform_kwargs)

    def _transform_base(self, **kwargs):
        return MolDesc(**kwargs)
    
class BaseGraph:

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def make_graph(self, mol: Chem.Mol):
        raise NotImplementedError
    
    def __call__(self, X: Chem.Mol|list[Chem.Mol]):

        if isinstance(X, Chem.Mol):
            X = [X]

        out = {}
        pbar = tqdm(total=len(X), desc='Generating graphs', disable=not self.verbose)
        for idx, mol in enumerate(X):
            if not isinstance(mol, Chem.Mol):
                out[idx] = None
            else:
                out[idx] = self.make_graph(mol)
            pbar.update(1)
        pbar.close()
        return out
    