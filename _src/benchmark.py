from .utils import get_model, get_tokenizer
from .data.utils import load_dataset
from .data.datasets import BaseDataset

from lightgbm import LGBMRegressor
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    SelectKBest, RFE, RFECV, VarianceThreshold,
    mutual_info_regression, mutual_info_classif
)
from tqdm import tqdm
from typing import Generator
import time
from datetime import datetime
from .utils import all_benchmarks

class CoCorr:
    """
    Fetaure selection based on collinearity.

    For highly correlated features, the feature with the lowest variance is removed.
    """
    def __init__(self, threshold: float = 0.9):
        self.threshold = threshold
        self.to_keep = []

    def fit(self, X: np.ndarray):
        var = X.std(axis=0)
        corr_matrix = np.corrcoef(X, rowvar=False)
        upper = np.triu(corr_matrix, k=1)
        idx = np.where(upper>0.9)
        idx = zip(*idx)
        exclude = []
        for i, j in idx:
            out = i if var[i] < var[j] else j
            exclude.append(int(out))
        self.to_keep = [i for i in range(X.shape[1]) if i not in set(exclude)]

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.to_keep]

    def fit_transform(self, X: np.ndarray):
        self.fit(X)
        return self.transform(X)

def rfecv(
        X_train: np.ndarray, y_train: np.ndarray, estimator,
        return_selector: bool = False, cv: int = 5,
    ) -> np.ndarray|tuple[np.ndarray, RFECV]:
    """
    Recursive Feature Elimination with X-validation.
    """
    rfe_selector = RFECV(estimator=estimator, cv=cv)
    X_train = rfe_selector.fit_transform(X_train, y_train)
    if return_selector:
        return X_train, rfe_selector
    else:
        return X_train
    
def rfe(
        X_train: np.ndarray, y_train: np.ndarray, estimator,
        n_features: int|float, return_selector: bool = False, 
    ) -> np.ndarray|tuple[np.ndarray, RFE]:
    rfe_selector = RFE(estimator=estimator, n_features_to_select=n_features)
    X_train = rfe_selector.fit_transform(X_train, y_train)
    if return_selector:
        return X_train, rfe_selector
    else:
        return X_train
    
def cocorr(X_train: np.ndarray, threshold: float = 0.9, return_selector: bool = True):
    """
    Remove highly correlated features.
    """
    cc = CoCorr(threshold=threshold)
    X_train = cc.fit_transform(X_train)
    if return_selector:
        return X_train, cc
    else:
        return X_train
    
def kbest(
    X_train: np.ndarray, y_train: np.ndarray, k: int,
    return_selector: bool = True, task: str = 'regression'
    ):
    """
    Select the k best features.
    """
    if task == 'classification':
        score_func = mutual_info_classif
    elif task == 'regression':
        score_func = mutual_info_regression
    else:
        raise ValueError('Invalid task. Must be either "classification" or "regression"')
    selector = SelectKBest(score_func=score_func, k=k)
    X_train = selector.fit_transform(X_train, y_train)
    if return_selector:
        return X_train, selector
    else:
        return X_train
    
class SelectAll:

    def fit(self, X: np.ndarray):
        pass

    def transform(self, X: np.ndarray):
        return X

    def fit_transform(self, X: np.ndarray):
        return X
    
def select_all(X_train: np.ndarray, return_selector: bool = True, **kwargs):
    if return_selector:
        return X_train, SelectAll()
    else:
        return X_train

def variance_threshold(
        X_train: np.ndarray, threshold: float = 0.0, return_selector: bool = False
    ):
    """
    Remove zero variance features.
    """
    vt = VarianceThreshold(threshold=threshold)
    X_train = vt.fit_transform(X_train)
    if return_selector:
        return X_train, vt
    else:
        return X_train

def benchmark(config: dict):

    name: str = config['name']
    data_path: str = config['data']
    results_path: str = config['results']
    verbose: bool = config['verbose']
    benchmark_data: list[str]|str = config['benchmark']
    model_kwargs = config.get('model_kwargs', {})
    model_class = get_model(config['model'])
    
    tokenizer_class = get_tokenizer(config['tokenizer'])
    transform_kwargs = config.get('transform_kwargs', {})
    pbar = tqdm(total=len(benchmark_data), desc='Benchmarking', disable=not verbose)
    for benchmark in benchmark_data:
        print(f'Benchmarking on {benchmark}') if verbose else None
        print(f'Loading benchmark {benchmark}') if verbose else None
        df: BaseDataset = load_dataset(
            name=benchmark, root=data_path, compression=True,
            verbose=verbose,
        )
        print(f'Task type: {df.task}') if verbose else None
        splits: Generator = df.splits
        num_splits = df.num_splits
        print(f'Number of splits: {num_splits}') if verbose else None
        print('Loading molecules') if verbose else None
        mols = df.rdkit_mols
        y = df.y.to_numpy()

        tokenizer = tokenizer_class(X=mols, y=y, transform_kwargs=transform_kwargs, verbose=verbose)

        out = np.zeros((num_splits, len(df)))
        kbar = tqdm(total=num_splits, desc='Splits', disable=not verbose)
        for idx, (train, test) in enumerate(splits):
            print(f'Processing split {idx}.') if verbose else None
            tokenizer.reset(train, test)
            train_X, train_y = tokenizer.train
            print(f'Original shape: {train_X.shape}.') if verbose else None
            print('Applying variance threshold.') if verbose else None
            train_X, var_selector = variance_threshold(train_X, return_selector=True)
            print('Applying cocorrelation feature selection.') if verbose else None
            train_X, cocorr_selector = cocorr(train_X, return_selector=True)
            k = train_X.shape[0] - 1
            if train_X.shape[1] > k:
                print(
                    f'More features than N - 1.\n\
                    Applying kbest feature selection with k = {k}.'
                ) if verbose else None
                train_X, kbest_selector = kbest(
                    train_X, train_y, k=k,
                    return_selector=True, task=df.task
                )
            else:
                print(
                    'Fewer features than N - 1, selecting all features.'
                ) if verbose else None
                train_X, kbest_selector = select_all(train_X, return_selector=True)
            print(f'Final shape: {train_X.shape}.') if verbose else None

            model = model_class(task=df.task, **model_kwargs)
            model.fit(train_X, train_y)
            train_pred = model.predict(train_X)
            out[idx, train] = train_pred
            test_X, _ = tokenizer.test
            test_X = var_selector.transform(test_X)
            test_X = cocorr_selector.transform(test_X)
            test_X = kbest_selector.transform(test_X)

            test_pred = model.predict(test_X)
            out[idx, test] = test_pred
            kbar.update(1)
        kbar.close()
        out_path = Path(results_path) / name
        out_path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path / f'{benchmark.lower()}_preds.npz', out)
        pbar.update(1)

    pbar.close()


