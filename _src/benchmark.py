from .utils import get_model, get_tokenizer
from .data.utils import load_dataset
from .data.datasets import BaseDataset
from .tokenizers.base import BaseTokenizer

from copy import deepcopy

import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    SelectKBest, RFE, RFECV, VarianceThreshold,
    mutual_info_regression, mutual_info_classif
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, average_precision_score
from tqdm import tqdm
from typing import Callable, Generator
import yaml

import optuna


class CoCorr:
    """
    Fetaure selection based on collinearity.

    For highly correlated features, the feature with the lowest variance is removed.

    Parameters
    ----------
    threshold : float
        The threshold for collinearity. Default is 0.9.

    Attributes
    ----------
    threshold : float
        The threshold for collinearity.
    to_keep : list
        The indices of the features to keep.
    """
    def __init__(self, threshold: float = 0.9):
        self.threshold = threshold
        self.to_keep = []

    def fit(self, X: np.ndarray):
        """
        Determine which features to keep based on multilinearity and variance.

        Parameters
        ----------
        X : np.ndarray
            The input data. The shape is (n_samples, n_features).
        """
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
        """
        Transform the input data by removing highly correlated features.

        Parameters
        ----------
        X : np.ndarray
            The input data. The shape is (n_samples, n_features).

        Returns
        -------
        np.ndarray
            The transformed data. The shape is (n_samples, len(self.to_keep)).
        """
        return X[:, self.to_keep]

    def fit_transform(self, X: np.ndarray):
        """
        Fit CoCorr to X and return the transformed X.

        Parameters
        ----------
        X : np.ndarray
            The input data. The shape is (n_samples, n_features).

        Returns
        -------
        np.ndarray
            The transformed data. The shape is (n_samples, len(self.to_keep)).
        """
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
    """
    Recursive Feature Elimination.

    Good for feature selection but slow.
    """
    rfe_selector = RFE(estimator=estimator, n_features_to_select=n_features)
    X_train = rfe_selector.fit_transform(X_train, y_train)
    if return_selector:
        return X_train, rfe_selector
    else:
        return X_train
    
def cocorr(
        X_train: np.ndarray,
        threshold: float = 0.9,
        return_selector: bool = True,
    ):
    """
    Remove highly correlated features. Finds pairs of features with a pearson correlation above the threshold.
    The feature with the lowest variance out of the pair is removed.

    Parameters
    ----------
    X_train : np.ndarray
        The input data. The shape is (n_samples, n_features).
    threshold : float
        The threshold for collinearity. Default is 0.9.
    return_selector : bool
        Whether to return the selector object to use with other arrays.
        Default is False.

    Returns
    -------
    np.ndarray
        The transformed data. The shape is (n_samples, n_features - n_removed).
    Optional[CoCorr]
        The selector object.
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
    Select the k best features using SelectKBest from sklearn.
    The score function is mutual information for both classification and regression tasks.

    Parameters
    ----------
    X_train : np.ndarray
        The input data. The shape is (n_samples, n_features).
    y_train : np.ndarray
        The target data. The shape is (n_samples,).
    k : int
        The number of features to select.
    return_selector : bool
        Whether to return the selector object to use with other arrays.
        Default is False.
    task : str
        The task type. Must be either 'classification' or 'regression'.

    Returns
    -------
    np.ndarray
        The transformed data. The shape is (n_samples, k).
    Optional[SelectKBest]
        The selector object.
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
    """
    Dummy class to select all features.
    """
    def fit(self, X: np.ndarray):
        pass

    def transform(self, X: np.ndarray):
        return X

    def fit_transform(self, X: np.ndarray):
        return X
    
def select_all(X_train: np.ndarray, return_selector: bool = True, **kwargs):
    """
    Wrapper for SelectAll class.
    """
    if return_selector:
        return X_train, SelectAll()
    else:
        return X_train

def variance_threshold(
        X_train: np.ndarray, threshold: float = 0.0, return_selector: bool = False
    ):
    """
    Remove low variance features. Default removes features with zero variance.

    Parameters
    ----------
    X_train : np.ndarray
        The input data. The shape is (n_samples, n_features).
    threshold : float
        The threshold for variance. Default is 0.0.
    return_selector : bool
        Whether to return the selector object to use with other arrays.
        Default is False.

    Returns
    -------
    np.ndarray
        The transformed data. The shape is (n_samples, n_features - n_removed).
    Optional[VarianceThreshold]
        The selector object.
    """
    vt = VarianceThreshold(threshold=threshold)
    X_train = vt.fit_transform(X_train)
    if return_selector:
        return X_train, vt
    else:
        return X_train

class HyperOpt:

    def __init__(
        self, model, model_kwargs: dict,
        task: str, hyperparameters: dict, tokenizer: BaseTokenizer,
        splits: list, scorer: Callable,
        direction: str = 'minimize', val_size: float = 0.2
    ):
        self.model = model
        self.model_kwargs = model_kwargs
        self.model_kwargs['verbose'] = -1
        self.tokenizer = tokenizer
        self.splits = splits
        self.scorer = scorer
        self.direction = direction
        self.hyperparameters = hyperparameters
        self.task = task
        self.val_size = val_size

    def objective(self, trial: optuna.Trial):
        hyperparameters = self.hyperparameters
        params = {}
        for key, value in hyperparameters.items():
            try:
                target = value['target']
            except:
                raise KeyError(
                    f'No distribution provided for hyperparameter {key}.'
                )
            value = {k: v for k, v in value.items() if k != 'target'}
            if target == 'int':
                p = trial.suggest_int(key, **value)
            elif target == 'float':
                p = trial.suggest_float(key, **value)
            elif target == 'categorical':
                p = trial.suggest_categorical(key, **value)
            else:
                raise ValueError(
                    'Invalid target. Must be one of "int", "float", or "categorical".'
                )
            params[key] = p

        params.update(self.model_kwargs)
        filler = np.inf if self.direction == 'minimize' else -np.inf
        out = np.full((len(self.splits,)), filler)
        for idx, (train, test) in enumerate(self.splits):
            self.tokenizer.reset(train, test)
            X, y = self.tokenizer.train
            train_X, val_X, train_y, val_y = train_test_split(
                X, y, test_size=self.val_size, random_state=42
            )
            train_X, var_selector = variance_threshold(train_X, return_selector=True)
            train_X, cocorr_selector = cocorr(train_X, return_selector=True)
            k = train_X.shape[0] - 1
            if train_X.shape[1] > k:
                train_X, kbest_selector = kbest(
                    train_X, train_y, k=k,
                    return_selector=True, task=self.task
                )
            else:
                train_X, kbest_selector = select_all(train_X, return_selector=True)
            model = self.model(task=self.task, **params)
            
            model.fit(train_X, train_y)
            
            val_X = var_selector.transform(val_X)
            val_X = cocorr_selector.transform(val_X)
            val_X = kbest_selector.transform(val_X)

            test_pred = model.predict(val_X)
            out[idx] = self.scorer(val_y, test_pred)
        return out.mean()
    
    def run(self, trials: int = 50):
        study = optuna.create_study(direction=self.direction)
        study.optimize(self.objective, n_trials=trials)
        return study.best_params

def benchmark(config: dict):
    """
    Benchmark a model on a list of datasets.

    Parameters
    ----------
    config : dict
        The configuration dictionary. Must have the following
        keys:
            - name: str
                The name of the model for saving predictions.
            - data: str
                The path to the data.
            - results: str
                The path to save the results.
            - verbose: bool
                Whether to print verbose output.
            - benchmark: list[str]|str
                The list of benchmarks to use.
            - model: str
                The model to use.
            - model_kwargs: dict
                The keyword arguments for the model.
            - tokenizer: str
                The tokenizer (i.e. molecular featurization method) to use.
            - transform_kwargs: dict
                The keyword arguments for the tokenizer.
    """
    name: str = config['name']
    data_path: str = config['data']
    results_path: str = config['results']
    verbose: bool = config['verbose']
    benchmark_data: list[str]|str = config['benchmark']
    model_class = get_model(config['model'])
    base_model_kwargs = config.get('model_kwargs', {})
    
    tokenizer_class = get_tokenizer(config['tokenizer'])
    transform_kwargs = config.get('transform_kwargs', {})

    hyperparameters: dict = config.get('model_hyperparameters', {})
    print(f'Hyperparameters: {hyperparameters}') if verbose else None
    tuning: bool = hyperparameters.pop(
        'tuning'
    ) if 'tuning' in hyperparameters else False
    trials: int = hyperparameters.pop(
        'trials'
    ) if 'trials' in hyperparameters else 50
    num_hyp_splits = hyperparameters.pop(
        'num_splits'
    ) if 'num_splits' in hyperparameters else 5

    pbar = tqdm(total=len(benchmark_data), desc='Benchmarking', disable=not verbose)
    for benchmark in benchmark_data:
        model_kwargs = {**base_model_kwargs}
        print(f'Benchmarking on {benchmark}') if verbose else None
        print(f'Loading benchmark {benchmark}') if verbose else None
        df: BaseDataset = load_dataset(
            name=benchmark, root=data_path, compression=True,
            verbose=verbose,
        )
        print(f'Task type: {df.task}') if verbose else None
        splits: list = list(df.splits)
        num_splits = df.num_splits
        print(f'Number of splits: {num_splits}') if verbose else None
        print('Loading molecules') if verbose else None
        mols = df.rdkit_mols
        y = df.y.to_numpy()

        tokenizer = tokenizer_class(X=mols, y=y, transform_kwargs=transform_kwargs, verbose=verbose)

        if f'{benchmark}_hyperparameters' in config:
            best_params = config[f'{benchmark}_hyperparameters']
            print(
                f'Using saved hyperparameters: \n{best_params}'
            ) if verbose else None
            model_kwargs.update(config[f'{benchmark}_hyperparameters'])
        
        else:
            print('No hyperparameters for benchmark found.') if verbose else None
        
            if tuning:
                print('Running hyperparameter tuning.') if verbose else None
                if df.task == 'regression':
                    scorer = mean_absolute_error
                    direction = 'minimize'
                else:
                    scorer = average_precision_score
                    direction = 'maximize'
                hyperopt_splits = [splits[i] for i in range(num_hyp_splits)]
                opt = HyperOpt(
                    model=model_class, model_kwargs=model_kwargs, task=df.task,
                    hyperparameters=hyperparameters, tokenizer=tokenizer,
                    splits=hyperopt_splits, scorer=scorer, direction=direction
                )
                best_params = opt.run(trials=trials)
                print(f'Best hyperparameters: {best_params}') if verbose else None

                current_config = yaml.load(open(config['path']), Loader=yaml.Loader)
                current_config |= {f'{benchmark}_hyperparameters': best_params}
                with open(config['path'], 'w') as f:
                    yaml.dump(current_config, f)
                print('Saved hyperparameters.') if verbose else None

            else:
                print('Using default hyperparameters.') if verbose else None

        out = np.zeros((num_splits, len(df)))
        kbar = tqdm(total=num_splits, desc='Splits', disable=not verbose)
        for idx, (train, test) in enumerate(splits):
            print('\n') if verbose else None
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
                    f'More features than N - 1.\
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


