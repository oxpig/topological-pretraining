from ...tokenizers import (
    BaseTokenizer,
    ECFP, FCFP, PDV, SNS,
    AtomGraphTokenizer, MorganGraphTokenizer, SNSGraphTokenizer
)
from ..feature_selection import CoCorr, SelectAll

from pathlib import Path
import numpy as np
from rdkit import Chem

from sklearn.feature_selection import (
    SelectKBest, VarianceThreshold,
    mutual_info_classif, mutual_info_regression,
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
import torch
import torch_geometric as pyg
from tqdm import tqdm
from typing import Callable, Literal, Optional

tokenizers_dict = {
    'ECFP': ECFP,
    'FCFP': FCFP,
    'PDV': PDV,
    'SNS': SNS,
    'AtomGraphTokenizer': AtomGraphTokenizer,
    'MorganGraphTokenizer': MorganGraphTokenizer,
    'SNSGraphTokenizer': SNSGraphTokenizer,
}

extra_transform_classes = {
            'variance_threshold': VarianceThreshold,
            'cocorr': CoCorr,
            'select_k_best': SelectKBest,
            'scaler': MinMaxScaler,
        }

class MolDataset:
    """
    Molecular dataset for training and testing on benchmark datasets.

    Parameters:
    -----------
    mols: list[Chem.Mol]
        List of RDKit molecules
    y: np.ndarray, optional
        Target values
    train_idx: np.ndarray, optional
        Indices of training samples
    test_idx: np.ndarray, optional
        Indices of test samples
    tokenizer: Literal[
        'ECFP', 'FCFP', 'PDV', 'SNS',
        'AtomGraphTokenizer', 'MorganGraphTokenizer', 'SNSGraphTokenizer'
    ], optional
        Tokenizer to use for featurization. Default is None.
        If None, self.X is a list of RDKit molecules.
    tokenizer_kwargs: dict, optional
        Keyword arguments for the tokenizer.
    extra_transform_kwargs: dict, optional
        Keyword arguments for extra transforms after tokenization.
        Currently supported transforms are:
        - variance_threshold (sklearn.feature_selection.VarianceThreshold)
        - cocorr (Filter features with high correlation)
        - select_k_best (sklearn.feature_selection.SelectKBest)
        - scaler (sklearn.preprocessing.MinMaxScaler)
        Input format: {transform name: {kwargs}}
        Default is {}.
    verbose: bool, optional
    """

    _extra_transform = SelectAll()

    def __init__(
        self,
        mols: list[Chem.Mol], y: Optional[np.ndarray] = None,
        train_idx: Optional[np.ndarray] = None, test_idx: Optional[np.ndarray] = None,
        tokenizer: Literal[
            'ECFP', 'FCFP', 'PDV', 'SNS',
            'AtomGraphTokenizer', 'MorganGraphTokenizer', 'SNSGraphTokenizer' 
        ] = None,
        tokenizer_kwargs: dict = {}, extra_transform_kwargs: dict = {},
        verbose: bool = False, fit_transform: bool = False,
    ):
        self.mols = mols
        self.X = None
        self.y = y
        self.tokenizer = tokenizer
        self.tokenizer_kwargs = tokenizer_kwargs
        self.set_extra_transform(extra_transform_kwargs)

        if self.tokenizer is not None:
            self.tokenizer: BaseTokenizer = tokenizers_dict[self.tokenizer]
            self.tokenizer = self.tokenizer(verbose=verbose, **self.tokenizer_kwargs)

        if train_idx is None:
            train_idx = np.arange(len(mols))

        if fit_transform:
            self.reset(train_idx, test_idx)

    def set_extra_transform(self, kwargs):
        """
        Set extra transforms after tokenization.
        Currently does not support transforms over graphs.

        Parameters:
        -----------
        kwargs: dict
            Names and keyword arguments for extra transforms.

        Sets self._extra_transform as a sklearn.pipeline.Pipeline object.
        """
        self.extra_transform_kwargs = kwargs
        _extra_transform = []
        for name, vals in kwargs.items():
            if name not in extra_transform_classes:
                raise ValueError(f'Invalid extra transform name: {name}')
            extra_transform = extra_transform_classes[name]
            _extra_transform.append((name, extra_transform(**vals)))
        if len(_extra_transform) > 0:
            self._extra_transform = Pipeline(_extra_transform)
    
    def __len__(self):
        return len(self.mols)

    def __getitem__(self, idx):
        """
        Get the featurized representation of a molecule.
        """
        X = self.X[idx] # Get the tokenized representation of the molecule
        if isinstance(X, np.ndarray):
            # If the representation is a 1D array, reshape it to a 2D array
            if len(X.shape) == 1:
                X = X.reshape(-1, 1)

        if self.y is None:
            return self._extra_transform.transform(X)
        else:
            return self._extra_transform.transform(X), self.y[idx]
    
    def reset(self, train_idx: np.ndarray, test_idx = None) -> None:
        """
        Refit the tokenizer and extra transforms with new training data
        and reset the train and test indices.

        Parameters:
        -----------
        train_idx: np.ndarray
            Indices of training samples
        test_idx: np.ndarray, optional
            Indices of test samples

        Sets self.train_idx and self.test_idx as the input indices.
        """
        self.train_idx, self.test_idx = train_idx, test_idx
        # Fit the tokenizer if it is not precomputed or if the tokenized data is not available
        # Precomputed tokenizers do not need to be refit (e.g. Morgan fingerprints)
        if not self.tokenizer.precomputed or self.X is None:
            train_mols = [self.mols[i] for i in train_idx]
            y = self.y[train_idx] if self.y is not None else None
            self.tokenizer.fit(train_mols, y)
            self.X = self.tokenizer.transform(self.mols) 
        
        # Set k to the number of samples - 1 for SelectKBest
        if 'selectkbest' in self._extra_transform.named_steps:
            self._extra_transform.set_params(selectkbest__k=self.X.shape[0] - 1)
        
        # Fit the extra transforms 
        y = self.y[train_idx] if self.y is not None else None
        train_X = [self[i] for i in train_idx]
        self._extra_transform.fit(train_X, y)


    @property
    def train_X(self):
        # Get the tokenized representation of the training data
        if isinstance(self.X, np.ndarray|torch.Tensor):
            train_X = self.X[self.train_idx]
        else:
            train_X = [self[i] for i in self.train_idx]
        
        # Apply the extra transforms
        train_X = self._extra_transform.transform(train_X)
        return train_X

    @property
    def train_y(self):
        if self.y is None:
            return
        else:
            return self.y[self.train_idx]
    
    @property
    def train(self):
        if self.y is None:
            return self.train_X
        else:
            return self.train_X, self.train_y
    
    @property
    def test_X(self):
        if self.test_idx is None:
            return None
        # Get the tokenized representation of the training data
        if isinstance(self.X, np.ndarray|torch.Tensor):
            test_X = self.X[self.test_idx]
        else:
            test_X = [self[i] for i in self.test_idx]
            
        # Apply the extra transforms
        test_X = self._extra_transform.transform(test_X)
        return test_X
    
    @property
    def test_y(self):
        if self.test_idx is None:
            return None
        elif self.y is None:
            return None
        else:
            return self.y[self.test_idx]
    
    @property
    def test(self):
        if self.y is None:
            return self.test_X
        else:
            return self.test_X, self.test_y

class GraphDataset(torch.utils.data.Dataset):

    def __init__(self, graphs: list[pyg.data.Data]):
        super(GraphDataset, self).__init__()
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx]