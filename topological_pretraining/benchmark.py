from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Literal

import numpy as np
import optuna
import torch
import yaml
from sklearn.base import BaseEstimator
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import mean_absolute_error, roc_auc_score
from tqdm import tqdm

from topological_pretraining.data import available_datasets, load_dataset
from topological_pretraining.data.datasets import BaseDataFrame, MolDataset
from topological_pretraining.logging import Logger
from topological_pretraining.models import LGBM, get_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""
Script for benchmarking models on various datasets.
"""


class HyperOpt:
	"""
	Hyperparameter optimization of scratch GIN and lightGBMs using Optuna on benchmark datasets.

	Parameters
	----------
	model : BaseEstimator
	    The model to optimize.
	model_kwargs : dict
	    The keyword arguments for the model, i.e., fixed hyperparameters.
	task : str
	    The task to optimize for. Must be one of 'classification' or 'regression'.
	hyperparameters : dict
	    The hyperparameters to optimize. Must be a dictionary with keys as hyperparameter names and values
	    as dictionaries with keys 'target' (str, one of 'int', 'float', 'categorical') and other parameters
	    for the Optuna suggestion method (e.g., 'low', 'high', 'choices').
	dataset : MolDataset
	    The dataset to use for optimization.
	splits : list
	    The list of splits to use for optimization. Each split is a tuple of (train_idx, val_idx).
	scorer : Callable
	    The scoring function to use for optimization. Must take two arguments: true values and predicted values.
	direction : str
	    The direction of optimization. Must be one of 'minimize' or 'maximize'.
	val_size : float
	    The size of the validation set. Only used if `splits` is not provided.
	verbose : bool
	    Whether to print verbose output.
	logging : Logger
	    A logger instance to store logging information such as losses and hyperparameters. Default is None.
	    Functions like a dictionary with additional saving and loading methods.
	name : str
	    The name of the hyperparameter tuning run. Used for logging.
	seed : int
	    The random seed to use for reproducibility.
	data_clusters : np.ndarray, optional
	    The data clusters to use for grouped cross-validation. If provided, must be a 1D array of integers,
	    where each integer corresponds to a cluster label for a sample in the dataset.
	average : Literal['mean', 'median']
	    The method to use for averaging the scores across splits. Must be one of 'mean' or 'median'.
	    Default is 'mean'.
	"""

	trial_count = 0

	def __init__(
		self,
		model: BaseEstimator,
		model_kwargs: dict,
		task: str,
		hyperparameters: dict,
		dataset: MolDataset,
		splits: list,
		scorer: Callable,
		direction: str = 'minimize',
		val_size: float = 0.2,
		verbose: bool = False,
		logging: Logger | None = None,
		name: str = 'name',
		seed: int = 42,
		data_clusters: np.ndarray = None,
		average: Literal['mean', 'median'] = 'mean',
	):
		self.seed = seed
		self.model = model
		self.logging = logging
		self.name = name
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
		self.data_clusters = data_clusters
		self.average = average
		if dataset.y is None:
			raise ValueError(
				'Dataset must have labels for hyperparameter optimization.'
			)
		if verbose == 2:
			optuna.logging.set_verbosity(optuna.logging.DEBUG)
		elif verbose == 1:
			optuna.logging.set_verbosity(optuna.logging.INFO)
		else:
			optuna.logging.set_verbosity(optuna.logging.WARNING)

	def objective(self, trial: optuna.Trial):
		"""
		The objective function for Optuna hyperparameter optimization.

		Parameters
		----------
		trial : optuna.Trial
		    The Optuna trial object containing the hyperparameters to evaluate.

		Returns
		-------
		float
		    The score for the current set of hyperparameters. This is the value that Optuna will
		    try to minimize or maximize based on the `direction` specified during initialization.
		"""
		hyperparameters = self.hyperparameters
		params = {}
		for key, value in hyperparameters.items():
			target = value.get('target')
			if target is None:
				raise KeyError(f'No distribution provided for hyperparameter {key}.')
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
		if self.logging is not None:
			for key, value in params.items():
				path_name = f'{self.name}/tuning_parameters/{key}'
				if path_name not in self.logging:
					self.logging[path_name] = []
				self.logging[path_name].append(value)
			self.logging.save()

		filler = np.inf if self.direction == 'minimize' else -np.inf
		out = np.full(
			(
				len(
					self.splits,
				)
			),
			filler,
		)

		for idx, (train_idx, val_idx) in enumerate(self.splits):
			self.dataset.reset(train_idx, val_idx)
			train_X, train_y = self.dataset.train
			val_X, val_y = self.dataset.test
			if self.model.__name__ == 'SklearnGIN':
				params['vocab_size'] = self.dataset.featurizer.vocab_size
				params['input_dim'] = train_X[0].x.size(1)
				params['name'] = f'{self.name}/trials'
				params['logging'] = self.logging
			model = self.model(seed=self.seed, task=self.task, **params)
			model.fit(train_X, train_y)
			test_pred = model.predict(val_X)

			out[idx] = self.scorer(val_y, test_pred)
			if self.logging is not None:
				tuning_score_path = f'{self.name}/tuning_scores'
				trial_num_path = f'{self.name}/trial_num_for_scores'
				if tuning_score_path not in self.logging:
					self.logging[tuning_score_path] = []
				if trial_num_path not in self.logging:
					self.logging[trial_num_path] = []
				self.logging[f'{self.name}/tuning_scores'].append(out[idx])
				self.logging[f'{self.name}/trial_num_for_scores'].append(
					self.trial_count
				)

		if self.average == 'mean':
			out_score = out.mean()
		elif self.average == 'median':
			out_score = np.median(out)
		else:
			raise ValueError('Invalid average. Must be one of "mean" or "median".')
		if self.logging is not None:
			if f'{self.name}/tuning_averages' not in self.logging:
				self.logging[f'{self.name}/tuning_averages'] = []
			self.logging[f'{self.name}/tuning_averages'].append(out_score)
			self.logging.save()

		self.trial_count += 1
		return out_score

	def run(self, trials: int = 50):
		"""
		Run the hyperparameter tuning using Optuna.

		Parameters
		----------
		trials : int
		    The number of trials to run for hyperparameter tuning. Default is 50.

		Returns
		-------
		tuple
		    A tuple containing the best hyperparameters found and the trial number of the best trial.
		"""
		print(
			f'Running hyperparameter tuning with {trials} trials.'
		) if self.verbose else None
		study = optuna.create_study(
			direction=self.direction,
			sampler=optuna.samplers.TPESampler(seed=42),
		)
		study.optimize(self.objective, n_trials=trials)
		return study.best_params, study.best_trial.number


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
	        - featurizer: str
	            The featurizer (i.e. molecular featurization method) to use.
	        - transform_kwargs: dict
	            The keyword arguments for the featurizer.
	"""
	# Setup configuration
	to_log = config.pop('logging', False)
	name: str = config['name']
	data_path: str = config['data']
	results_path: str = config['results']
	verbose: bool = config['verbose']
	benchmark_data: list[str] | str = config['benchmark']
	if benchmark_data == 'biogen':
		benchmark_data = [
			'Human_PPB',
			'Rat_PPB',
			'Solu',
			'Efflux',
			'Human_CLint',
			'Rat_CLint',
		]
	elif benchmark_data == 'subset':
		benchmark_data = [
			'Human_PPB',
			'Rat_PPB',
			'Solu',
			'Efflux',
			'Human_CLint',
			'Rat_CLint',
			'Lipo',
			'ESOL',
			'FreeSolv',
			'DRD2',
			'FactorXA',
			'MUV466',
			'MUV548',
			'MUV600',
			'NR_AR',
			'NR_ER_LBD',
			'SR_ARE',
			'SR_p53',
		]
	elif benchmark_data == 'all':
		benchmark_data = [i for i in available_datasets if i != 'QMugs']
	elif isinstance(benchmark_data, list):
		for benchmark in benchmark_data:
			if benchmark not in available_datasets:
				raise ValueError(
					f'Benchmark {benchmark} not found. Available benchmarks: {available_datasets}'
				)
	if isinstance(benchmark_data, str):
		benchmark_data = [benchmark_data]

	model_class = get_model(config['model'])
	base_model_kwargs = config.get('model_kwargs', {})
	base_model_kwargs['device'] = device
	featurizer_class = config['featurizer']
	featurizer_kwargs = config.get('featurizer_kwargs', {})
	extra_transform_kwargs = config.get('extra_transform_kwargs', {})
	clusters = config.get('clusters')

	seed = config.get('seed', 42)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)

	print('\n##################################################\n') if verbose else None
	print(f'Run {name}.') if verbose else None
	print(f'Benchmarking {config["featurizer"]}.') if verbose else None
	print(f'Featurizer kwargs: {featurizer_kwargs}') if verbose else None

	hyperparameters: dict = config.get('model_hyperparameters', {})
	print(f'Hyperparameters: {hyperparameters}') if verbose else None
	tuning: bool = (
		hyperparameters.pop('tuning') if 'tuning' in hyperparameters else False
	)
	trials: int = hyperparameters.pop('trials') if 'trials' in hyperparameters else 50
	num_hyp_splits = (
		hyperparameters.pop('num_splits') if 'num_splits' in hyperparameters else 5
	)

	hyperparam_path = Path(data_path) / 'hyperparameters'
	if 'experiment' in config:
		hyperparam_path = hyperparam_path / config['experiment']
	hyperparam_path = hyperparam_path / name
	hyperparam_path.mkdir(parents=True, exist_ok=True)

	# loop over benchmark datasets
	pbar = tqdm(total=len(benchmark_data), desc='Benchmarking', disable=not verbose)
	for benchmark in benchmark_data:
		if to_log:
			logging = Logger()
			logging['config'] = config
			logging.path = Path(results_path) / name / f'{benchmark}_logging.npz'
			logging.path.parent.mkdir(parents=True, exist_ok=True)
			logging.save()
		else:
			logging = None
		hyperparameters_running = deepcopy(hyperparameters)
		model_kwargs = {**base_model_kwargs}
		print(
			'\n##################################################\n'
		) if verbose else None
		print(f'Benchmarking on {benchmark}') if verbose else None
		print(f'Loading benchmark {benchmark}') if verbose else None
		df: BaseDataFrame = load_dataset(
			name=benchmark,
			root=data_path,
			compression=True,
			verbose=verbose,
		)
		if (
			not config.get('adjust_hyperparameters')
			and model_class.__name__ == 'SklearnGIN'
		):
			print('Using default HP search space') if verbose else None
		elif len(df) < 500 and model_class.__name__ == 'SklearnGIN':
			print(
				'Fewer than 500 data points. Setting GIN HP search space'
			) if verbose else None
			if 'node_embedding_dim' in hyperparameters_running:
				hyperparameters_running['node_embedding_dim']['choices'] = [
					2,
					4,
					8,
				]
			hyperparameters_running['hidden_dim']['choices'] = [
				4,
				8,
			]
			hyperparameters_running['head_hidden_dim']['choices'] = [4, 8]
			hyperparameters_running['head_layers']['high'] = 3
			base_model_kwargs['batch_size'] = 16

		elif 500 <= len(df) < 1000 and model_class.__name__ == 'SklearnGIN':
			print(
				'Fewer than 1000 data points. Setting GIN HP search space'
			) if verbose else None
			if 'node_embedding_dim' in hyperparameters_running:
				hyperparameters_running['node_embedding_dim']['choices'] = [
					4,
					8,
					16,
				]
			hyperparameters_running['hidden_dim']['choices'] = [
				4,
				8,
				16,
			]
			hyperparameters_running['head_hidden_dim']['choices'] = [
				4,
				8,
				16,
			]
			base_model_kwargs['batch_size'] = 32

		elif 1000 <= len(df) < 5000 and model_class.__name__ == 'SklearnGIN':
			print(
				'Between 1000 and 5000 data points. Setting GIN HP search space'
			) if verbose else None
			if 'node_embedding_dim' in hyperparameters_running:
				hyperparameters_running['node_embedding_dim']['choices'] = [
					8,
					16,
					32,
				]
			hyperparameters_running['hidden_dim']['choices'] = [
				8,
				16,
				32,
			]
			hyperparameters_running['head_hidden_dim']['choices'] = [
				8,
				16,
				32,
			]
			base_model_kwargs['batch_size'] = 64

		elif len(df) >= 5000 and model_class.__name__ == 'SklearnGIN':
			print(
				'More than 5000 data points. Setting GIN HP search space'
			) if verbose else None
			if 'node_embedding_dim' in hyperparameters_running:
				hyperparameters_running['node_embedding_dim']['choices'] = [
					16,
					32,
					64,
				]
			hyperparameters_running['hidden_dim']['choices'] = [
				16,
				32,
				64,
			]
			hyperparameters_running['head_hidden_dim']['choices'] = [
				16,
				32,
				64,
			]
			base_model_kwargs['batch_size'] = 128
		else:
			print('Using default HP search space') if verbose else None

		splits: list = list(df.splits)
		num_splits = df.num_splits
		if df.task == 'regression':
			scorer = mean_absolute_error
			direction = 'minimize'
		else:
			scorer = roc_auc_score
			direction = 'maximize'

		if logging is not None:
			logging['num_splits'] = num_splits
			logging['task'] = df.task
			logging['scorer'] = str(scorer)
			logging['direction'] = direction
			logging['hp_average'] = df.hyperopt_average

		out_path = Path(results_path) / name
		out_path.mkdir(parents=True, exist_ok=True)
		out = np.zeros((num_splits, len(df) + 1))
		if config['model'] == 'SklearnGIN':
			lgbm_out = np.zeros((num_splits, len(df) + 1))
		complete = {}
		if (out_path / f'{benchmark.lower()}_preds.npz').exists():
			print('Predictions already exist. Getting checkpoint.') if verbose else None
			try:
				preds = np.load(out_path / f'{benchmark.lower()}_preds.npz')
				out = preds['arr_0']
				complete = {i: True for i in np.where(out[:, -1] == 1)[0]}
			except (OSError, ValueError, KeyError) as e:
				print(
					f'Error: {e}\n'
					f'Error loading {benchmark.lower()} predictions '
					f'for {name}. Starting from scratch.'
				) if verbose else None

		if out[-1, -1] == 1:
			print('All splits complete.') if verbose else None
			if logging is not None:
				logging['train_score'] = []
				logging['test_score'] = []
				print('Calculating scores.') if verbose else None
				for idx, (train, test) in enumerate(splits):
					train_y = df.y[train]
					test_y = df.y[test]
					train_pred = out[idx, train]
					test_pred = out[idx, test]
					train_score = scorer(train_y, train_pred)
					test_score = scorer(test_y, test_pred)
					logging['train_score'].append(train_score)
					logging['test_score'].append(test_score)
				print('Scores calculated and logged.') if verbose else None
			print('Skipping to next benchmark.') if verbose else None

			if config['model'] == 'SklearnGIN':
				if (out_path / f'{benchmark.lower()}_lgbm_preds.npz').exists():
					try:
						lgbm_out = np.load(
							out_path / f'{benchmark.lower()}_lgbm_preds.npz'
						)['arr_0']
					except (OSError, ValueError, KeyError) as e:
						print(
							f'Error: {e}\n'
							f'Error loading LGBM predictions for {benchmark.lower()}: {e}'
						) if verbose else None
				if lgbm_out[-1, -1] == 1:
					if logging is not None:
						logging['lgbm_train_score'] = []
						logging['lgbm_test_score'] = []
					for idx, (train, test) in enumerate(splits):
						train_y = df.y[train]
						test_y = df.y[test]
						train_pred = lgbm_out[idx, train]
						test_pred = lgbm_out[idx, test]
						train_score = scorer(train_y, train_pred)
						test_score = scorer(test_y, test_pred)
						if logging is not None:
							logging['lgbm_train_score'] = []
							logging['lgbm_test_score'] = []
							logging['lgbm_train_score'].append(train_score)
							logging['lgbm_test_score'].append(test_score)

			pbar.update(1)
			continue

		print(f'Task type: {df.task}') if verbose else None
		print(f'Number of splits: {num_splits}') if verbose else None
		print('Loading molecules') if verbose else None
		mols = df.rdkit_mols
		y = df.y.to_numpy()

		if 'select_k_best' in extra_transform_kwargs:
			extra_transform_kwargs['select_k_best']['score_func'] = {
				'regression': mutual_info_regression,
				'classification': mutual_info_classif,
			}[df.task]

		dataset = MolDataset(
			mols=mols,
			y=y,
			featurizer=featurizer_class,
			featurizer_kwargs=featurizer_kwargs,
			extra_transform_kwargs=extra_transform_kwargs,
			verbose=verbose,
			fit_transform=False,
		)

		print('Dataset loaded.') if verbose else None
		print('Checking for saved hyperparameters.') if verbose else None
		benchmark_hp_path = hyperparam_path / f'{benchmark}.yaml'
		best_trial_num = None
		if benchmark_hp_path.exists():
			with open(benchmark_hp_path) as f:
				best_params = yaml.safe_load(f)
			print(f'Using saved hyperparameters: \n{best_params}') if verbose else None
			best_trial_num = best_params.pop('best_trial')
			model_kwargs.update(best_params)

		else:
			print('No hyperparameters for benchmark found.') if verbose else None

			if tuning:
				print('Running hyperparameter tuning.') if verbose else None
				hyperopt_splits = [splits[i] for i in range(num_hyp_splits)]
				data_clusters = None
				if clusters is not None:
					if clusters not in df.columns:
						print(
							'Clusters not found in dataframe. Skipping using clusters for hyperopt'
						) if verbose else None
					else:
						data_clusters = df[clusters]
						data_clusters = data_clusters.to_numpy().astype(int)
				opt = HyperOpt(
					model=model_class,
					model_kwargs=model_kwargs,
					task=df.task,
					hyperparameters=hyperparameters_running,
					dataset=dataset,
					splits=hyperopt_splits,
					scorer=scorer,
					direction=direction,
					verbose=verbose,
					logging=logging,
					name=f'{benchmark}_hyperopt',
					seed=seed,
					average=df.hyperopt_average,
				)
				best_params, best_trial_num = opt.run(trials=trials)
				model_kwargs.update(best_params)
				print(f'Best hyperparameters: {best_params}') if verbose else None
				print(f'Best trial: {best_trial_num}') if verbose else None
				best_params['best_trial'] = best_trial_num
				with open(benchmark_hp_path, 'w') as f:
					yaml.dump(best_params, f)
				print('Saved hyperparameters.') if verbose else None

			else:
				print('Using default hyperparameters.') if verbose else None
		if logging is not None:
			logging['model_kwargs'] = model_kwargs
			logging['best_trial'] = best_trial_num
			logging.save()
		kbar = tqdm(
			total=num_splits,
			desc=f'Benchmark: {benchmark} | Splits',
			disable=not verbose,
		)
		if logging is not None:
			logging['train_score'] = []
			logging['test_score'] = []
		for idx, (train, test) in enumerate(splits):
			if idx in complete:
				# If the split is already complete, skip to the next split. This allows for resuming from checkpoints.
				train_y = df.y[train]
				test_y = df.y[test]
				train_pred = out[idx, train]
				test_pred = out[idx, test]
				train_score = scorer(train_y, train_pred)
				test_score = scorer(test_y, test_pred)
				if logging is not None:
					logging['train_score'].append(train_score)
					logging['test_score'].append(test_score)
				kbar.update(1)
				continue
			print('\n') if verbose == 2 else None
			print(f'Processing split {idx}.') if verbose == 2 else None
			dataset.reset(train, test)
			train_X, train_y = dataset.train
			if isinstance(train_X, np.ndarray):
				print(
					f'Train shape: {dataset.train_X.shape}.'
				) if verbose == 2 else None

			if verbose == 2:
				model_kwargs['verbose'] = 1
			else:
				model_kwargs['verbose'] = -1

			if config['model'] == 'SklearnGIN':
				model_kwargs['verbose'] = 0
				model_kwargs['input_dim'] = train_X[0].x.size(1)
				model_kwargs['vocab_size'] = dataset.featurizer.vocab_size

			print('Fitting model...') if verbose else None
			model = model_class(
				seed=seed, task=df.task, logging=logging, **model_kwargs
			)

			model.fit(train_X, train_y)
			print('Getting predictions...') if verbose else None
			train_pred = model.predict(train_X)
			out[idx, train] = train_pred

			test_X, _ = dataset.test

			test_pred = model.predict(test_X)
			out[idx, test] = test_pred
			out[idx, -1] = 1  # Mark as complete
			if config['model'] == 'SklearnGIN':
				train_embeddings = model.embed(train_X)
				lgbm_model = LGBM(task=df.task, seed=seed, verbose=-1)
				lgbm_model.fit(train_embeddings, train_y)
				lgbm_train_pred = lgbm_model.predict(train_embeddings)
				lgbm_out[idx, train] = lgbm_train_pred
				test_embeddings = model.embed(test_X)
				lgbm_test_pred = lgbm_model.predict(test_embeddings)
				lgbm_out[idx, test] = lgbm_test_pred

			if logging is not None:
				test_score = scorer(df.y[test], test_pred)
				train_score = scorer(df.y[train], train_pred)
				logging['train_score'].append(train_score)
				logging['test_score'].append(test_score)
				if config['model'] == 'SklearnGIN':
					lgbm_test_score = scorer(df.y[test], lgbm_test_pred)
					lgbm_train_score = scorer(df.y[train], lgbm_train_pred)
					logging['lgbm_train_score'].append(lgbm_train_score)
					logging['lgbm_test_score'].append(lgbm_test_score)
				logging.save()

			np.savez_compressed(out_path / f'{benchmark.lower()}_preds.npz', out)
			if config['model'] == 'SklearnGIN':
				np.savez_compressed(
					out_path / f'{benchmark.lower()}_lgbm_preds.npz', lgbm_out
				)
			del model
			kbar.update(1)
		kbar.close()

		pbar.update(1)
		del dataset

	pbar.close()
