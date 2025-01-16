import numpy as np
from rdkit import Chem

class BaseTokenizer:

    def __init__(
        self,
        X: list[Chem.Mol],
        y: np.ndarray,
        train: np.ndarray = np.array([]),
        test: np.ndarray = np.array([]),
        transform: callable = lambda x: x,
    ):
        self.transform = transform
        self.X = self.transform(X)
        self.y = y
        self.train_idx = train
        self.test_idx = test

    def reset(self, train: np.ndarray, test: np.ndarray) -> None:
        self.train_idx = train
        self.test_idx = test

    @property
    def train(self):
        return self.X[self.train_idx], self.y[self.train_idx]
    
    @property
    def test(self):
        return self.X[self.test_idx], self.y[self.test_idx]