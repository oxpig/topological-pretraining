from _src.tokenizers import (
    BaseTokenizer,
    ECFP, FCFP, PDV, SNS,
    AtomGraphTokenizer, MorganGraphTokenizer,
    PreTrainedTokenizer
)
from _src.data.feature_selection import CoCorr, SelectAll

from pathlib import Path
import numpy as np
from rdkit import Chem

from sklearn.feature_selection import (
    SelectKBest, VarianceThreshold,
    mutual_info_classif, mutual_info_regression,
)
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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
    'PreTrainedTokenizer': PreTrainedTokenizer,
}

extra_transform_classes = {
    'variance_threshold': VarianceThreshold,
    'cocorr': CoCorr,
    'select_k_best': SelectKBest,
    'pca': PCA,
    'minmax_scaler': MinMaxScaler,
    'standard_scaler': StandardScaler,
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
    imputer = None

    def __init__(
        self,
        mols: list[Chem.Mol], y: Optional[np.ndarray] = None,
        train_idx: Optional[np.ndarray] = None, test_idx: Optional[np.ndarray] = None,
        tokenizer: Literal[
            'ECFP', 'FCFP', 'PDV', 'SNS',
            'AtomGraphTokenizer', 'MorganGraphTokenizer'
        ] = None,
        tokenizer_kwargs: dict = {}, extra_transform_kwargs: dict = {},
        verbose: bool = False, fit_transform: bool = False,
    ):
        self.verbose = verbose
        self.mols = mols
        self.X = None
        self.y = y
        self.tokenizer = tokenizer
        self.tokenizer_kwargs = tokenizer_kwargs
        self.set_extra_transform(extra_transform_kwargs)

        if self.tokenizer is not None:
            print('Setting tokenizer...') if self.verbose else None
            if self.tokenizer not in tokenizers_dict:
                raise ValueError(f'Invalid tokenizer: {self.tokenizer}')
            self.tokenizer: BaseTokenizer = tokenizers_dict[self.tokenizer]
            self.tokenizer = self.tokenizer(verbose=verbose, **self.tokenizer_kwargs)
            print(f'Tokenizer set.\n{self.tokenizer}') if self.verbose else None

        if train_idx is None:
            train_idx = np.arange(len(mols))
            test_idx = np.array([])

        if self.tokenizer is not None:
            print('Preprocessing molecules...') if self.verbose else None
            self.mols = self.tokenizer.preprocess(self.mols)

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
        print('Setting extra transforms...') if self.verbose else None
        self.extra_transform_kwargs = kwargs
        _extra_transform = []
        for name, vals in kwargs.items():
            if name not in extra_transform_classes:
                raise ValueError(f'Invalid extra transform name: {name}')
            extra_transform = extra_transform_classes[name]
            if name == 'select_k_best':
                assert 'score_func' in vals, 'score_func must be provided for SelectKBest'
            _extra_transform.append((name, extra_transform(**vals)))
        if len(_extra_transform) > 0:
            self._extra_transform = Pipeline(_extra_transform)
        print(f'Extra transforms set.\nPipeline: {self._extra_transform}') if self.verbose else None
    
    def __len__(self):
        return len(self.mols)

    def __getitem__(self, idx):
        """
        Get the featurized representation of a molecule.
        """
        print(idx)
        X = self.X[idx] # Get the tokenized representation of the molecule
        if isinstance(X, np.ndarray):
            # If the representation is a 1D array, reshape it to a 2D array
            if len(X.shape) == 1:
                X = X.reshape(1,-1)

        if self.imputer is not None:
            X = self.imputer.transform(X)

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
        print('Resetting train and test indices...') if self.verbose else None
        self.train_idx, self.test_idx = train_idx, test_idx
        # Fit the tokenizer if it is not precomputed or if the tokenized data is not available
        # Precomputed tokenizers do not need to be refit (e.g. Morgan fingerprints)
        if not self.tokenizer.precomputed or self.X is None:
            print('Fitting tokenizer...') if self.verbose else None
            train_mols = [self.mols[i] for i in train_idx]
            y = self.y[train_idx] if self.y is not None else None
            self.tokenizer.fit(train_mols, y)
            self.X = self.tokenizer.transform(self.mols) 
        else:
            print('Tokenizer is precomputed. Skipping fit.') if self.verbose else None
        if isinstance(self.X, np.ndarray):
            if np.any(np.isnan(self.X)):
                num_cols = len(np.where(np.isnan(self.X).sum(axis=0))[0])
                num_rows = len(np.where(np.isnan(self.X).sum(axis=1))[0])
                num_nans = np.isnan(self.X).sum()
                print(f'NaN values found in X; number of cols = {num_cols}; number of rows = {num_rows}; total NaNs = {num_nans}.')
                print('Fitting imputer on train data for replacing NaN values with mean...')
                self.imputer = SimpleImputer(strategy='mean')
                self.imputer.fit(self.X[train_idx])
        else:
            self.imputer = None

        # Set k to the number of samples - 1 for SelectKBest
        if 'select_k_best' in self._extra_transform.named_steps:
            k = self.train_idx.shape[0]
            print(f'Setting k for SelectKBest to number of train samples - 1...') if self.verbose else None
            self._extra_transform.set_params(select_k_best__k=k)
            print(f'k set to {k}.') if self.verbose else None
        
        # Fit the extra transforms 
        print('Fitting extra transforms...') if self.verbose else None
        y = self.y[train_idx] if self.y is not None else None
        train_X = [self.X[i] for i in train_idx]
        if self.imputer is not None:
            train_X = self.imputer.transform(train_X)
        self._extra_transform.fit(train_X, y)

    @property
    def train_X(self):
        return self[self.train_idx][0]

    @property
    def train_y(self):
        self[self.train_idx][1]
    
    @property
    def train(self):
        return self[self.train_idx]
    
    @property
    def test_X(self):
        if self.test_idx is None:
            return None
        else:
            return self[self.test_idx][0]
    
    @property
    def test_y(self):
        if self.test_idx is None:
            return None
        else:
            return self[self.test_idx][1]
    
    @property
    def test(self):
        if self.test_idx is None:
            return None
        else:
            return self[self.test_idx]
