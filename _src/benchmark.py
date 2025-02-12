from .utils import get_model, get_tokenizer
from .data.utils import load_dataset
from .data.datasets import BaseDataFrame, MolDataset
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

class HyperOpt:

    def __init__(
        self, model, model_kwargs: dict,
        task: str, hyperparameters: dict, dataset: MolDataset,
        splits: list, scorer: Callable,
        direction: str = 'minimize', val_size: float = 0.2,
        verbose: bool = False
    ):
        self.model = model
        self.model_kwargs = model_kwargs
        self.model_kwargs['verbose'] = -1
        self.dataset = dataset
        self.splits = splits
        self.scorer = scorer
        self.direction = direction
        self.hyperparameters = hyperparameters
        self.task = task
        self.val_size = val_size
        self.verbose = verbose
        if verbose == 2:
            optuna.logging.set_verbosity(optuna.logging.DEBUG)
        elif verbose == 1:
            optuna.logging.set_verbosity(optuna.logging.INFO)
        else:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

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
            self.dataset.reset(train_idx, val_idx)

            train_X, train_y = self.dataset.train
            val_X, val_y = self.dataset.test
            model = self.model(seed=42, task=self.task, **params)
            model.fit(train_X, train_y)
            test_pred = model.predict(val_X)
            out[idx] = self.scorer(val_y, test_pred)

        return out.mean()
    
    def run(self, trials: int = 50):
        print(f'Running hyperparameter tuning with {trials} trials.') if self.verbose else None
        study = optuna.create_study(direction=self.direction, sampler=optuna.samplers.TPESampler(seed=42), )
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

    tokenizer_class = config['tokenizer']
    tokenizer_kwargs = config.get('tokenizer_kwargs', {})
    extra_transform_kwargs = config.get('extra_transform_kwargs', {})

    print(f'\n##################################################\n') if verbose else None
    print(f'Run {name}.') if verbose else None
    print(f'Benchmarking {config["tokenizer"]}.') if verbose else None
    print(f'Tokenizer kwargs: {tokenizer_kwargs}') if verbose else None

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

    hyperparam_path = Path(data_path) / f'hyperparameters/{name}'
    hyperparam_path.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(total=len(benchmark_data), desc='Benchmarking', disable=not verbose)
    for benchmark in benchmark_data:
        model_kwargs = {**base_model_kwargs}
        print(f'\n##################################################\n') if verbose else None
        print(f'Benchmarking on {benchmark}') if verbose else None
        
        print(f'Loading benchmark {benchmark}') if verbose else None
        df: BaseDataFrame = load_dataset(
            name=benchmark, root=data_path, compression=True,
            verbose=verbose,
        )
        splits: list = list(df.splits)
        num_splits = df.num_splits
        out_path = Path(results_path) / name
        out_path.mkdir(parents=True, exist_ok=True)
        out = np.zeros((num_splits, len(df) + 1))
        complete = {}
        if (out_path / f'{benchmark.lower()}_preds.npz').exists():
            print('Predictions already exist. Getting checkpoint.') if verbose else None
            preds = np.load(out_path / f'{benchmark.lower()}_preds.npz')
            out = preds['arr_0']
            complete = {i: True for i in np.where(out[:, -1] == 1)[0]}
        if out[-1, -1] == 1:
            print('All splits complete. Skipping') if verbose else None
            pbar.update(1)
            continue
        print(f'Task type: {df.task}') if verbose else None
        
        print(f'Number of splits: {num_splits}') if verbose else None
        print('Loading molecules') if verbose else None
        mols = df.rdkit_mols
        y = df.y.to_numpy()

        dataset = MolDataset(
            mols=mols, y=y,
            tokenizer=tokenizer_class, tokenizer_kwargs=tokenizer_kwargs,
            extra_transform_kwargs=extra_transform_kwargs,
            verbose=verbose, fit_transform=False,
        )
    
        print('Dataset loaded.') if verbose else None
        print('Checking for saved hyperparameters.') if verbose else None
        benchmark_hp_path = hyperparam_path / f'{benchmark}.pt'
        if benchmark_hp_path.exists():
            best_params = torch.load(benchmark_hp_path, map_location='cpu')
            print(
                f'Using saved hyperparameters: \n{best_params}'
            ) if verbose else None
            model_kwargs.update(best_params)
        
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
                    hyperparameters=hyperparameters, dataset=dataset,
                    splits=hyperopt_splits, scorer=scorer, direction=direction,
                    verbose=verbose
                )
                best_params = opt.run(trials=trials)
                print(f'Best hyperparameters: {best_params}') if verbose else None

                torch.save(best_params, benchmark_hp_path)
                print('Saved hyperparameters.') if verbose else None

            else:
                print('Using default hyperparameters.') if verbose else None
        kbar = tqdm(total=num_splits, desc='Splits', disable=not verbose)
        for idx, (train, test) in enumerate(splits):
            if idx in complete:
                kbar.update(1)
                continue
            print('\n') if verbose == 2 else None
            print(f'Processing split {idx}.') if verbose == 2 else None
            dataset.reset(train, test)
            print(f'Train shape: {dataset.train_X.shape}.') if verbose == 2 else None

            if verbose == 2:
                model_kwargs['verbose'] = 1
            else:
                model_kwargs['verbose'] = -1
            model = model_class(seed=42, task=df.task, **model_kwargs)
            train_X, train_y = dataset.train
            model.fit(train_X, train_y)
            train_pred = model.predict(train_X)
            out[idx, train] = train_pred

            test_X, _ = dataset.test

            test_pred = model.predict(test_X)
            out[idx, test] = test_pred
            out[idx, -1] = 1 # Mark as complete
            np.savez_compressed(out_path / f'{benchmark.lower()}_preds.npz', out)

            kbar.update(1)
        kbar.close()
        
        pbar.update(1)

    pbar.close()


