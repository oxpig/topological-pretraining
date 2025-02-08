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
import torch
from tqdm import tqdm
from typing import Callable, Generator
import yaml

import optuna




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
    pass
    
    
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
    



class HyperOpt:

    def __init__(
        self, model, model_kwargs: dict,
        task: str, hyperparameters: dict, tokenizer: BaseTokenizer,
        splits: list, scorer: Callable,
        direction: str = 'minimize', val_size: float = 0.2,
        verbose: bool = False
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
        self.verbose = verbose

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
            train_idx, val_idx = train_test_split(
                train, test_size=self.val_size, random_state=42, shuffle=True
            )

            self.tokenizer.reset(train_idx, val_idx)
            self.tokenizer.set_variance_threshold()
            self.tokenizer.set_cocorr()
            k = self.tokenizer.train_X.shape[0] - 1
            if self.tokenizer.train_X.shape[1] > k:
                self.tokenizer.set_select_k_best(k=k, task=self.task)
            else:
                pass
            self.tokenizer.set_min_max_scale()
            train_X, train_y = self.tokenizer.train
            val_X, val_y = self.tokenizer.test
            model = self.model(seed=42, task=self.task, **params)
            model.fit(train_X, train_y)
            val_X, _ = self.tokenizer.test
            test_pred = model.predict(val_X)
            out[idx] = self.scorer(val_y, test_pred)
        print(out)
        return out.mean()
    
    def run(self, trials: int = 50):
        print(f'Running hyperparameter tuning with {trials} trials.') if self.verbose else None
        study = optuna.create_study(direction=self.direction, sampler=optuna.samplers.TPESampler(seed=42))
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

    hyperparam_path = Path(data_path) / f'results/hyperparameters/{name}'
    hyperparam_path.mkdir(parents=True, exist_ok=True)

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

        tokenizer: BaseTokenizer = tokenizer_class(
            X=mols, y=y, transform_kwargs=transform_kwargs,
            verbose=verbose
        )

        benchmark_hp_path = hyperparam_path / f'{benchmark}.pt'
        if benchmark_hp_path.exists():
            best_params = torch.load(benchmark_hp_path, map_location='cpu', allow_pickle=True)
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
                    splits=hyperopt_splits, scorer=scorer, direction=direction,
                    verbose=verbose
                )
                best_params = opt.run(trials=trials)
                print(f'Best hyperparameters: {best_params}') if verbose else None

                torch.save(best_params, benchmark_hp_path)
                print('Saved hyperparameters.') if verbose else None

            else:
                print('Using default hyperparameters.') if verbose else None

        out = np.zeros((num_splits, len(df)))
        kbar = tqdm(total=num_splits, desc='Splits', disable=not verbose)
        for idx, (train, test) in enumerate(splits):
            print('\n') if verbose else None
            print(f'Processing split {idx}.') if verbose else None
            tokenizer.reset(train, test)
            # train_X, train_y = tokenizer.train
            print(f'Original shape: {tokenizer.train_X.shape}.') if verbose else None
            print('Applying variance threshold.') if verbose else None
            tokenizer.set_variance_threshold()
            print(f'New shape: {tokenizer.train_X.shape}.') if verbose else None
            # train_X, var_selector = variance_threshold(train_X, return_selector=True)
            print('Applying cocorrelation feature selection.') if verbose else None
            # train_X, cocorr_selector = cocorr(train_X, return_selector=True)
            tokenizer.set_cocorr()
            print(f'New shape: {tokenizer.train_X.shape}.') if verbose else None
            k = tokenizer.train_X.shape[0] - 1
            if tokenizer.train_X.shape[1] > k:
                print(
                    f'More features than N - 1.\
                    Applying kbest feature selection with k = {k}.'
                ) if verbose else None
                tokenizer.set_select_k_best(k=k, task=df.task)
                # print(f'New shape: {tokenizer.train_X.shape}.') if verbose else None
                # train_X, kbest_selector = kbest(
                #     train_X, train_y, k=k,
                #     return_selector=True, task=df.task
                # )
            else:
                print(
                    'Fewer features than N - 1, selecting all features.'
                ) if verbose else None
                # train_X, kbest_selector = select_all(train_X, return_selector=True)
            print(f'Final shape: {tokenizer.train_X.shape}.') if verbose else None
            print(f'Setting feature scaling.') if verbose else None
            tokenizer.set_min_max_scale()


            model = model_class(seed=42, task=df.task, **model_kwargs)
            train_X, train_y = tokenizer.train
            model.fit(train_X, train_y)
            train_pred = model.predict(train_X)
            out[idx, train] = train_pred

            test_X, _ = tokenizer.test

            test_pred = model.predict(test_X)
            out[idx, test] = test_pred
            kbar.update(1)
        kbar.close()
        out_path = Path(results_path) / name
        out_path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path / f'{benchmark.lower()}_preds.npz', out)
        pbar.update(1)

    pbar.close()


