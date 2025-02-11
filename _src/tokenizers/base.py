import numpy as np
from rdkit import Chem
import torch
import torch_geometric as pyg
from tqdm import tqdm
from typing import Callable, Optional
from sklearn.feature_selection import (
    SelectKBest, VarianceThreshold,
    mutual_info_classif, mutual_info_regression
)
from sklearn.preprocessing import MinMaxScaler

from copy import deepcopy

from ..data.mol import MolDesc
from ..data.feature_selection import CoCorr, SelectAll

class BaseTokenizer:

    _transform = None
    variance_threshold = SelectAll()
    cocorr = SelectAll()
    select_k_best = SelectAll()
    scaler = SelectAll()
    fixed_transform_kwargs = {}

    def __init__(
        self,
        transform_kwargs: dict = {},
        variance_threshold: Optional[float] = None,
        cocorr_threshold: Optional[float] = None,
        k_best_task: Optional[int] = None,
        scaler: bool = False,
        verbose: bool = False,
    ):
        self.verbose = verbose
        if isinstance(variance_threshold, float):
            self.variance_threshold = VarianceThreshold(threshold=variance_threshold)
        if isinstance(cocorr_threshold, float):
            self.cocorr = CoCorr(threshold=cocorr_threshold)
        if k_best_task == 'classification':
            self.k_best_func = mutual_info_classif
        elif k_best_task == 'regression':
            self.k_best_func = mutual_info_regression
        else:
            self.k_best_func = None

        if scaler:
            self.scaler = MinMaxScaler()
        transform_kwargs.update(self.fixed_transform_kwargs)
        self.set_transform(kwargs=transform_kwargs)


    def __call__(self, X: Chem.Mol|list[Chem.Mol]) -> np.ndarray:
        X = self.transform(X)
        X = self.variance_threshold.transform(X)
        X = self.cocorr.transform(X)
        X = self.select_k_best.transform(X)
        X = self.scaler.transform(X)
        return X

    @property
    def transform(self):
        return self._transform
    
    def set_transform(self, kwargs):
        self._transform = self._transform_base(**kwargs)

    def _transform_base(self, **kwargs):
        raise NotImplementedError

    def reset(self, mols: Chem.Mol) -> None:
        raise NotImplementedError
    
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

class BaseGraphTokenizer(BaseTokenizer):

    def __init__(
        self,
        transform_kwargs: dict = {},
        verbose: bool = False,
    ):
        super(BaseGraphTokenizer, self).__init__(
            transform_kwargs=transform_kwargs, verbose=verbose
        )

    @property
    def train_X(self):
        train_graphs = [deepcopy(self.X[i]) for i in self.train_idx]
        for i, graph in enumerate(train_graphs):
            x = graph.x
            x = self.variance_threshold.transform(x)
            x = self.cocorr.transform(x)
            x = self.scaler.transform(x)
            train_graphs[i].x = torch.tensor(x)
            edge_attr = graph.edge_attr
            if edge_attr is None:
                continue
            if edge_attr.shape[0] == 0:
                continue
            edge_attr = self.edge_attr_variance_threshold.transform(edge_attr)
            edge_attr = self.edge_attr_cocorr.transform(edge_attr)
            edge_attr = self.edge_attr_scaler.transform(edge_attr)
            train_graphs[i].edge_attr = torch.tensor(edge_attr)
        return train_graphs
    
    @property
    def train_y(self):
        return self.y[self.train_idx] if self.y is not None else None
    
    @property
    def test_X(self):
        test_graphs = [deepcopy(self.X[i]) for i in self.test_idx]
        for i, graph in enumerate(test_graphs):
            x = graph.x
            x = self.variance_threshold.transform(x)
            x = self.cocorr.transform(x)
            x = self.scaler.transform(x)
            test_graphs[i].x = torch.tensor(x)
            edge_attr = graph.edge_attr
            if edge_attr is None:
                continue
            if edge_attr.shape[0] == 0:
                continue
            edge_attr = self.edge_attr_variance_threshold.transform(edge_attr)
            edge_attr = self.edge_attr_cocorr.transform(edge_attr)
            edge_attr = self.edge_attr_scaler.transform(edge_attr)
            test_graphs[i].edge_attr = torch.tensor(edge_attr)
        return test_graphs

    @property
    def test_y(self):
        return self.y[self.test_idx] if self.y is not None else None

    @property
    def train(self):
        return self.train_X, self.train_y
    
    @property
    def test(self):
        return self.test_X, self.test_y
    
    def set_variance_threshold(self, threshold: float = 0.0) -> np.ndarray:
        """
        Remove features with low variance.
        
        Parameters
        ----------
        threshold : float
            Variance threshold.
        
        Returns
        -------
        np.ndarray
            Mask of features to keep.
        """
        self.variance_threshold = SelectAll()
        self.cocorr = SelectAll()

        X_train = pyg.data.Batch.from_data_list(self.train_X)

        if X_train.x.shape[1] == 1:
            return

        vt = VarianceThreshold(threshold=threshold)
        vt.fit(X_train.x)
        self.variance_threshold = vt

        vt = VarianceThreshold(threshold=threshold)
        if X_train.edge_attr.shape[1] == 1:
            return
        vt.fit(X_train.edge_attr)
        self.edge_attr_variance_threshold = vt

    def set_cocorr(self, threshold: float = 0.9):
        """
        Remove features with high cocorrelation.
        Finds pairs of features with a correlation higher than the threshold.
        The feature with the highest variance is kept.
        
        Parameters
        ----------
        threshold : float
            Correlation threshold.
        """
        self.cocorr = SelectAll()
        X_train = pyg.data.Batch.from_data_list(self.train_X)

        if X_train.x.shape[1] == 1:
            return

        cc = CoCorr(threshold=threshold)
        cc.fit(X_train.x)
        self.cocorr = cc

        cc = CoCorr(threshold=threshold)
        if X_train.edge_attr.shape[1] == 1:
            return
        cc.fit(X_train.edge_attr)
        self.edge_attr_cocorr = cc

    def set_select_k_best(self, score_func: Callable, k: int = 10):
        self.select_k_best = SelectAll()

    def set_min_max_scale(self):
        use = self.use_scaler
        self.scaler = SelectAll()
        if use:
            X_train = pyg.data.Batch.from_data_list(self.train_X)
            if X_train.x.shape[1] == 1:
                return
            scaler = MinMaxScaler()
            scaler.fit(X_train.x)
            self.scaler = scaler
            if X_train.edge_attr.shape[1] == 1:
                return
            scaler = MinMaxScaler()
            scaler.fit(X_train.edge_attr)
            self.edge_attr_scaler = scaler


    