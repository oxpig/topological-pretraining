from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from tqdm import tqdm

from topological_pretraining.data.datasets import BaseDataFrame, GraphDataset
from topological_pretraining.data.loader import DataLoader
from topological_pretraining.data.utils import load_dataset
from topological_pretraining.featurization import get_featurizer
from topological_pretraining.logging import Logger
from topological_pretraining.nn import get_nn
from topological_pretraining.nn.autoencoder import AutoEncoder
from topological_pretraining.nn.pred_head import (
	BinaryHead,
	MultiClassHead,
	MultiTaskLoss,
	RegressionHead,
)

"""
Script for pretraining models on graph datasets.
"""

pred_head_map = {
	'binary': BinaryHead,
	'regression': RegressionHead,
	'multiclass': MultiClassHead,
}

# TODO: change to handle multiple targets


def pretrain(config: dict):
	"""
	Run the pretraining process.

	Parameters:
	----------
	config : dict
	    Configuration dictionary containing the following keys:
	    - name: str
	        The name of the pretraining run.
	    - experiment: str
	        The name of the experiment.
	    - raw_name: str, optional
	        Name of the directory where the graphs are stored.
	        Defaults to the value of `experiment`.
	    - data: str
	        The path to the data directory.
	    - results: str
	        The path to the results directory.
	    - verbose: bool
	        Whether to print verbose output. Defaults to False.
	    - pretrain_data: list[str]|str
	        The name(s) of the dataset(s) to use for pretraining.
	    - featurizer: str
	        The class name of the featurizer to use.
	    - featurizer_kwargs: dict, optional
	        Additional keyword arguments for the featurizer.
	    - device: str
	        The device to use for training. Defaults to 'cuda' if available, otherwise 'cpu'.
	    - batch_size: int
	        The batch size for training. Defaults to 32.
	    - epochs: int
	        The number of epochs to train for. Defaults to 100.
	    - warmup_epochs: int, optional
	        The number of warmup epochs for learning rate scheduling. Defaults to 0.
	    - lr_decay_half_life: int, optional
	        The half-life for learning rate decay. Defaults to 5.
	    - weight_decay: float, optional
	        The weight decay for the optimizer. Defaults to 0.0.
	    - targets: dict
	        A dictionary of targets for the model, where each key is a target name and the value is a dictionary
	        containing target-specific parameters.
	    - splits: list[str], optional
	        A list of splits to use for training. If empty, all data will be used for pretraining.
	    - logging: bool, optional
	        Whether to use logging for the pretraining run. Defaults to False.
	        If True, a Logger object will be created and saved to the logging directory.
	    - seed: int
	        The random seed for reproducibility. Defaults to 42.
	    - model: str
	        The class name of the model to use for pretraining.
	    - model_kwargs: dict, optional
	        Additional keyword arguments for the model.
	    - standardization: bool, optional
	        Settings for standardizing the molecules.
	    - class_weights: bool, optional
	        Whether to use class weights for the loss function. Defaults to False.
	"""
	name: str = config['name']
	experiment_name: str = config['experiment']
	raw_name: str = config.get('raw_name', experiment_name)
	data_path: str = config['data']
	results_path: str = config['results']
	verbose: bool = config['verbose']
	pretrain_data: list[str] | str = config['pretrain_data']
	featurizer_class = config['featurizer']
	featurizer_kwargs = config.get('featurizer_kwargs', {})
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	batch_size: int = config['batch_size']
	epochs: int = config['epochs']
	warmup_epochs: int = config.get('warmup_epochs', 0)
	lr_decay_half_life: int = config.get('lr_decay_half_life', 5)
	weight_decay: float = config.get('weight_decay', 0.0)
	targets: dict = config['targets']
	splits: list[str] = config.get('splits', [])
	logging = config.get('logging', False)
	if logging:
		logging = Logger(
			path=Path(results_path) / name / f'{experiment_name}_pretrain_logging.npz'
		)
		logging['config'] = config
		logging.save()

	seed = config['seed']
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)

	model_class = get_nn(config['model'])
	model_kwargs = config.get('model_kwargs', {})
	if model_kwargs.get('graph_pool_type', None) == 'global_node':
		featurizer_kwargs['global_token'] = True
	else:
		featurizer_kwargs['global_token'] = False

	print('\n##################################################\n') if verbose else None
	print(f'Pretraining run {name}') if verbose else None
	print(f'Featurizer: {featurizer_class}') if verbose else None
	print(f'Model: {config["model"]}') if verbose else None
	print(f'Data: {pretrain_data}') if verbose else None
	print(f'Splits: {splits}') if verbose else None
	print(f'Batch size: {batch_size}') if verbose else None
	print(f'Epochs: {epochs}') if verbose else None
	print(f'Warmup epochs: {warmup_epochs}') if verbose else None
	print(f'Learning rate decay half life: {lr_decay_half_life}') if verbose else None
	print(f'Weight decay: {weight_decay}') if verbose else None
	print(f'Device: {device}') if verbose else None
	print('\n##################################################\n') if verbose else None

	# Load the dataset as a dataframe
	df: BaseDataFrame = load_dataset(
		name=pretrain_data, root=data_path, verbose=verbose
	)

	# Load the featurizer
	featurizer = get_featurizer(featurizer_class)(transform_kwargs=featurizer_kwargs)

	root = Path(data_path) / pretrain_data / raw_name
	root.mkdir(parents=True, exist_ok=True)

	if len(splits) == 0:
		print('No splits found, using all data for pretraining.') if verbose else None
		df['tanimoto_filter'] = 1
		splits = ['tanimoto_filter']

	if not config.get('standardization', True):
		print('Loading molecules without standardization...') if verbose else None
		mols = df.SMILES.to_list()
		mols = [Chem.MolFromSmiles(mol) for mol in mols]

	else:
		mols = df.rdkit_mols

	if isinstance(mols, np.ndarray):
		mols = mols.tolist()
	# prepare and save raw graphs
	raw_dataset = GraphDataset(
		root=root,
		featurizer=featurizer,
		molecules=mols,
		fit_featurizer=False,
		verbose=verbose,
	)
	del raw_dataset  # free up memory
	torch.cuda.empty_cache()
	save_path = results_path / experiment_name
	save_path.mkdir(parents=True, exist_ok=True)
	print(f'Looping through splits: {splits}') if verbose else None
	for split in splits:
		if split not in df.columns:
			raise ValueError(f'Split {split} not found in dataframe columns.')
		if len(splits) > 1:
			file_name = f'{name}_{split.replace(".", "")}.pt'
		else:
			file_name = f'{name}.pt'
		if (save_path / file_name).exists():
			print(f'Model {file_name} already exists, skipping.') if verbose else None
			continue
		idx = df[split]
		# load the dataset for this split
		pretrain_dataset = GraphDataset(
			root=root,
			split=(split, idx),
			featurizer=featurizer,
			targets=targets,
			run_id=name,
			fit_featurizer=True,
			verbose=verbose,
		)
		# prepare the dataloader for this split
		pretrain_loader = DataLoader(
			pretrain_dataset, batch_size=batch_size, shuffle=True
		)
		model_kwargs['input_dim'] = pretrain_dataset[0].x.size(1)
		if 'node_embedding' in model_kwargs and isinstance(
			model_kwargs['node_embedding'], int
		):
			model_kwargs['node_embedding'] = (
				len(pretrain_dataset.featurizer.node_types),
				model_kwargs['node_embedding'],
			)

		model = torch.nn.ModuleDict(
			{
				'main': model_class(device=device, **model_kwargs),
			}
		)
		model['main'].reset_parameters()
		model['main'].to(device)
		input_0 = pretrain_dataset[0].to(device)
		if model_kwargs.get('graph_pool_type', None) == 'global_node':
			if 'global_idx' not in input_0:
				raise ValueError('Global node pooling requires global node embeddings.')
			if input_0.global_idx is None:
				raise ValueError('Global node pooling requires global node embeddings.')

		print(f'Input 0: {input_0}') if verbose else None
		out = model['main'](**input_0)
		head_input_dim = out['global_state'].size(-1)
		targets_key = pretrain_dataset.targets

		is_regression = torch.empty((len(targets_key),), dtype=torch.bool)
		head_kwargs = {}

		for i, target_name in enumerate(targets_key):
			head_name = targets_key[target_name]['prediction_head']
			head_cls = pred_head_map[head_name]

			output_dim = input_0[target_name].size(1)
			class_weights = None
			print(f'Head: {head_name} | Output dim: {output_dim}') if verbose else None
			if head_name == 'multiclass' or head_name == 'binary':
				if config.get('class_weights', False):
					class_weights = targets_key[target_name]['class_weights'].to(device)
				is_regression[i] = False
			else:
				is_regression[i] = True

			head_kwargs[target_name] = {
				'input_dim': head_input_dim,
				'hidden_dim': model_kwargs.get('hidden_dim', 128),
				'output_dim': output_dim,
				'num_layers': 2,
				'act': model_kwargs.get('act', 'relu'),
				'dropout': model_kwargs.get('dropout', 0.0),
				'batch_norm': model_kwargs.get('batch_norm', False),
				'class_weights': class_weights,
			}

			head = head_cls(**head_kwargs[target_name])
			model[target_name] = head

		if len(is_regression) == 1:
			model['losses'] = torch.nn.Identity()

		else:
			model['losses'] = MultiTaskLoss(is_regression=is_regression)
		print(f'Model: {model}') if verbose else None
		num_params = sum(p.numel() for p in model.parameters())
		print(f'Number of parameters: {num_params}') if verbose else None
		if logging is not None:
			logging['model/num_params'] = num_params
			logging['model/num_heads'] = len(targets_key)
			logging['model/summary'] = str(model)
			logging.save()
		model = model.to(device)
		model.train()
		lr = num_params**-0.5
		if logging is not None:
			logging[f'{split}/lr'] = lr
			logging.save()
		optimizer = torch.optim.Adam(
			model.parameters(), lr=lr, weight_decay=weight_decay
		)
		warmup_steps = int(len(pretrain_loader) * warmup_epochs)
		steps_in_5_epochs = int(len(pretrain_loader) * lr_decay_half_life)
		gamma_for_halflife_5_epochs = 0.5 ** (1 / steps_in_5_epochs)
		scheduler = torch.optim.lr_scheduler.SequentialLR(
			optimizer=optimizer,
			milestones=[warmup_steps],
			schedulers=[
				torch.optim.lr_scheduler.LinearLR(
					optimizer,
					start_factor=0.5,
					end_factor=1.0,
					total_iters=warmup_steps,
				),
				torch.optim.lr_scheduler.ExponentialLR(
					optimizer, gamma=gamma_for_halflife_5_epochs
				),
			],
		)

		for epoch in range(epochs):
			epoch_loss = 0
			epoch_scores = torch.zeros(len(targets_key), device=device)
			pbar = tqdm(
				total=len(pretrain_loader),
				desc=f'Epoch {epoch + 1}/{epochs} | Batch Loss: {torch.nan}',
				disable=not verbose,
			)
			for batch_num, batch in enumerate(pretrain_loader):
				batch = batch.to(device)
				optimizer.zero_grad()
				out = model['main'](
					x=batch.x,
					edge_index=batch.edge_index,
					batch=batch.batch,
					global_idx=batch.get('global_idx'),
				)
				losses = torch.empty(len(targets_key), device=device)
				scores = torch.empty(len(targets_key), device=device)
				last_score = torch.tensor(0.0, device=device)
				for i, target_name in enumerate(targets_key):
					head = model[target_name]
					y = batch[target_name]
					# if y.layout != torch.strided:
					#     y = y.to_dense()
					if targets_key[target_name]['level'] == 'global':
						embed = out['global_state']
						embed.to(device)
						pred = head(embed)
						y = y.type(embed.dtype)
						losses[i] = head.loss(
							pred=pred,
							y=y,
						)
						if batch_num % 100 == 0:
							score_now = head.score(
								pred=pred,
								y=y,
							)
							last_score = deepcopy(score_now)
						else:
							score_now = None
					elif targets_key[target_name]['level'] == 'node':
						embed = out['final_state']
						embed.to(device)
						pred = head(embed)
						y = y.type(embed.dtype)
						mask = batch[f'{target_name}_mask'].type(embed.dtype)
						losses[i] = head.loss(
							pred=pred,
							y=y,
							mask=mask,
						)
						if batch_num % 100 == 0:
							score_now = head.score(
								pred=pred,
								y=y,
							)
							last_score = deepcopy(score_now)
						else:
							score_now = None
					else:
						raise ValueError('Invalid target level.')
					if score_now is not None:
						if logging is not None:
							score_path = f'{split}/batch_{target_name}_score'
							if score_path not in logging:
								logging[score_path] = []
							logging[score_path].append(score_now.item())
						scores[i] = score_now.item()
						epoch_scores[i] += score_now.item()

				loss = model['losses'](losses)
				loss.backward()
				optimizer.step()
				epoch_loss += loss.item()
				if logging is not None:
					if f'{split}/batch_loss' not in logging:
						logging[f'{split}/batch_loss'] = []
					logging[f'{split}/batch_loss'].append(loss.item())

				pbar.set_description(
					f'Epoch {epoch + 1}/{epochs} | Batch Loss: {loss.item():.4f} | Batch Score: {last_score.item():.4f}'
				)
				pbar.update(1)
				scheduler.step()

			average_loss = epoch_loss / len(pretrain_loader)
			average_scores = epoch_scores / len(pretrain_loader)
			pbar.set_description(
				f'Epoch {epoch + 1}/{epochs} | Epoch Loss: {average_loss:.4f} | Last Batch Loss: {loss.item():.4f} | Last Batch Score: {last_score.item():.4f}'
			)
			pbar.close()
			if logging is not None:
				if f'{split}/epoch_average_loss' not in logging:
					logging[f'{split}/epoch_average_loss'] = []
				logging[f'{split}/epoch_average_loss'].append(average_loss)
				for i, target_name in enumerate(targets_key):
					score_now = average_scores[i].item()
					if f'{split}/{target_name}_epoch_average_score' not in logging:
						logging[f'{split}/{target_name}_epoch_average_score'] = []
					logging[f'{split}/{target_name}_epoch_average_score'].append(
						score_now
					)
				logging.save()
			if (epoch + 1) % 15 == 0:
				model_dict = {
					'featurizer': pretrain_dataset.featurizer.to_dict(),
					'main': {
						'state': model['main'].state_dict(),
						'cls': model_class.__name__,
						'kwargs': model_kwargs,
					},
				}
				model_dict['heads'] = {}
				for _i, target_name in enumerate(targets_key):
					state = model[target_name].state_dict()
					head_name = targets_key[target_name]['prediction_head']
					head_cls = pred_head_map[head_name].__name__
					head_kwargs_to_save = head_kwargs[target_name]

					model_dict['heads'][target_name] = {
						'state': state,
						'cls': head_cls,
						'kwargs': head_kwargs_to_save,
						'level': targets_key[target_name]['level'],
					}

				torch.save(
					model_dict,
					results_path / experiment_name / f'epoch_{epoch + 1}_{file_name}',
				)

		model_dict = {
			'featurizer': pretrain_dataset.featurizer.to_dict(),
			'main': {
				'state': model['main'].state_dict(),
				'cls': model_class.__name__,
				'kwargs': model_kwargs,
			},
		}
		model_dict['heads'] = {}
		for _i, target_name in enumerate(targets_key):
			state = model[target_name].state_dict()
			head_name = targets_key[target_name]['prediction_head']
			head_cls = pred_head_map[head_name].__name__
			head_kwargs_to_save = head_kwargs[target_name]

			model_dict['heads'][target_name] = {
				'state': state,
				'cls': head_cls,
				'kwargs': head_kwargs_to_save,
			}

		torch.save(model_dict, results_path / experiment_name / file_name)
		Path(pretrain_dataset.processed_paths[0]).unlink()
		del pretrain_dataset
		del model
		del model_dict
		del pretrain_loader


def pretrain_autoencoder(config: dict):
	"""
	Run the pretraining process.
	Same parameters as `pretrain`, but for an autoencoder model rather than a GNN.
	"""
	name: str = config['name']
	experiment_name: str = config['experiment']
	raw_name: str = config.get('raw_name', experiment_name)
	data_path: str = config['data']
	results_path: str = config['results']
	verbose: bool = config['verbose']
	pretrain_data: list[str] | str = config['pretrain_data']
	featurizer_class = config['featurizer']
	featurizer_kwargs = config.get('featurizer_kwargs', {})
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	batch_size: int = config['batch_size']
	epochs: int = config['epochs']
	warmup_epochs: int = config.get('warmup_epochs', 0)
	lr_decay_half_life: int = config.get('lr_decay_half_life', 5)
	weight_decay: float = config.get('weight_decay', 0.0)

	splits: list[str] = config.get('splits', [])
	logging = config.get('logging', False)
	seed = config['seed']
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)

	model_class = AutoEncoder
	model_kwargs = config.get('model_kwargs', {})

	print('\n##################################################\n') if verbose else None
	print(f'Pretraining run {name}') if verbose else None
	print(f'Featurizer: {featurizer_class}') if verbose else None
	print(f'Model: {config["model"]}') if verbose else None
	print(f'Data: {pretrain_data}') if verbose else None
	print(f'Splits: {splits}') if verbose else None
	print(f'Batch size: {batch_size}') if verbose else None
	print(f'Epochs: {epochs}') if verbose else None
	print(f'Warmup epochs: {warmup_epochs}') if verbose else None
	print(f'Learning rate decay half life: {lr_decay_half_life}') if verbose else None
	print(f'Weight decay: {weight_decay}') if verbose else None
	print(f'Device: {device}') if verbose else None
	print('\n##################################################\n') if verbose else None

	# Load the dataset as a dataframe
	df: BaseDataFrame = load_dataset(
		name=pretrain_data, root=data_path, verbose=verbose
	)

	# Load the featurizer
	featurizer = get_featurizer(featurizer_class)(transform_kwargs=featurizer_kwargs)

	root = Path(data_path) / pretrain_data / raw_name
	root.mkdir(parents=True, exist_ok=True)

	if len(splits) == 0:
		print('No splits found, using all data for pretraining.') if verbose else None
		df['tanimoto_filter'] = 1
		splits = ['tanimoto_filter']

	if not config.get('standardization', True):
		print('Loading molecules without standardization...') if verbose else None
		mols = df.SMILES.to_list()
		mols = [Chem.MolFromSmiles(mol) for mol in mols]

	else:
		mols = df.rdkit_mols

	save_path = results_path / experiment_name
	save_path.mkdir(parents=True, exist_ok=True)

	print(f'Looping through splits: {splits}') if verbose else None
	for split in splits:
		if split not in df.columns:
			raise ValueError(f'{split} not found in dataframe.')
		file_name = f'{name}_{split}.pt' if len(splits) > 1 else f'{name}.pt'

		if (save_path / file_name).exists():
			print(f'Model {file_name} already exists, skipping.') if verbose else None
			continue
		idx = df[df[split] == 1].index
		split_mols = [mols[i] for i in idx]
		print(len(split_mols)) if verbose else None

		dataset_path = root / f'{split}.pt'
		if dataset_path.exists():
			dataset = torch.load(dataset_path, map_location=device, weights_only=True)
		else:
			featurizer.fit(split_mols)
			dataset = featurizer.transform(split_mols)
			if isinstance(dataset, np.ndarray):
				dataset = torch.tensor(dataset)
			torch.save(dataset, dataset_path)

		if model_kwargs.get('class_weights') is not None:
			if model_kwargs['decoder_type'] == 'multiclass':
				dataset_summed = dataset.sum(dim=0, dtype=torch.float64)
				dataset_summed[dataset_summed == 0] = (
					dataset_summed[dataset_summed == 0] + 1e-8
				)
				weights = torch.tensor(
					dataset.size(0) / (dataset.size(1) * dataset_summed)
				)
				weights.type(torch.float64)
			elif model_kwargs['decoder_type'] == 'binary':
				zero_class = torch.zeros_like(dataset, dtype=torch.float32)
				zero_class[dataset == 0] = 1
				zero_class_sum = zero_class.sum(dim=0, dtype=torch.float32)
				zero_class_sum[zero_class_sum == 0] = (
					zero_class_sum[zero_class_sum == 0] + 1
				)
				data_sum = dataset.sum(dim=0, dtype=torch.float32)
				data_sum[data_sum == 0] = data_sum[data_sum == 0] + 1
				weights = torch.stack(
					[
						dataset.size(0) / (2 * zero_class_sum),
						dataset.size(0) / (2 * data_sum),
					]
				)
				weights.type(torch.float32)
			else:
				raise ValueError('Invalid decoder type for using class weights.')
			model_kwargs['class_weights'] = weights

		dataset = torch.utils.data.TensorDataset(dataset)
		loader = torch.utils.data.DataLoader(
			dataset, batch_size=batch_size, shuffle=True
		)
		model_kwargs['input_dim'] = dataset[0][0].size(0)
		model = model_class(**model_kwargs)

		print(f'Model: {model}') if verbose else None
		num_params = sum(p.numel() for p in model.parameters())
		print(f'Number of parameters: {num_params}') if verbose else None
		if logging is not None:
			logging['model/num_params'] = num_params
			logging['model/summary'] = str(model)
		model = model.to(device)
		model.train()
		lr = num_params**-0.5
		if logging is not None:
			logging[f'{split}/lr'] = lr
		optimizer = torch.optim.Adam(
			model.parameters(), lr=lr, weight_decay=weight_decay
		)
		warmup_steps = int(len(loader) * warmup_epochs)
		steps_in_5_epochs = int(len(loader) * lr_decay_half_life)
		gamma_for_halflife_5_epochs = 0.5 ** (1 / steps_in_5_epochs)
		scheduler = torch.optim.lr_scheduler.SequentialLR(
			optimizer=optimizer,
			milestones=[warmup_steps],
			schedulers=[
				torch.optim.lr_scheduler.LinearLR(
					optimizer,
					start_factor=0.5,
					end_factor=1.0,
					total_iters=warmup_steps,
				),
				torch.optim.lr_scheduler.ExponentialLR(
					optimizer, gamma=gamma_for_halflife_5_epochs
				),
			],
		)
		for epoch in range(epochs):
			epoch_loss = 0
			epoch_scores = []
			last_score = np.nan
			pbar = tqdm(
				total=len(loader),
				desc=f'Epoch {epoch + 1}/{epochs} | Batch Loss: {torch.nan}',
				disable=not verbose,
			)
			for batch_num, batch in enumerate(loader):
				batch = batch[0]
				batch = batch.to(dtype=torch.float32, device=device)
				optimizer.zero_grad()
				pred = model.pred(x=batch)
				pred_vals = pred.detach().cpu().numpy()
				pred_vals = np.where(pred_vals > 0.5, 1, 0)
				if batch_num % 100 == 0:
					score = model.decoder.score(y=batch, pred=pred)
					if logging is not None:
						if f'{split}/batch_score' not in logging:
							logging[f'{split}/batch_score'] = []
						logging[f'{split}/batch_score'].append(score.item())
					last_score = score.item()
					epoch_scores.append(last_score)
				loss = model.decoder.loss(y=batch, pred=pred)
				loss.backward()
				optimizer.step()
				scheduler.step()
				epoch_loss += loss.item()
				if logging is not None:
					if f'{split}/batch_loss' not in logging:
						logging[f'{split}/batch_loss'] = []
					logging[f'{split}/batch_loss'].append(loss.item())
				pbar.set_description(
					f'Epoch {epoch + 1}/{epochs} | Batch Loss: {loss.item():.4f} | Batch Score: {last_score:.4f} | '
				)
				pbar.update(1)
			average_loss = epoch_loss / len(loader)
			average_score = sum(epoch_scores) / len(epoch_scores)
			pbar.set_description(
				f'Epoch {epoch + 1}/{epochs} | Epoch Loss: {average_loss:.4f} | Epoch Score: {average_score:.4f}'
			)
			pbar.close()

			if logging is not None:
				if f'{split}/epoch_average_loss' not in logging:
					logging[f'{split}/epoch_average_loss'] = []
				if f'{split}/epoch_average_score' not in logging:
					logging[f'{split}/epoch_average_score'] = []
				logging[f'{split}/epoch_average_loss'].append(average_loss)
				logging[f'{split}/epoch_average_score'].append(average_score)
				logging.save()

			if (epoch + 1) % 15 == 0:
				model_dict = {
					'featurizer': featurizer.to_dict(),
					'state': model.state_dict(),
					'cls': model_class.__name__,
					'kwargs': model_kwargs,
				}
				torch.save(
					model_dict,
					results_path / experiment_name / f'epoch_{epoch + 1}_{file_name}',
				)

		model_dict = {
			'featurizer': featurizer.to_dict(),
			'main': {
				'state': model.state_dict(),
				'cls': model_class.__name__,
				'kwargs': model_kwargs,
			},
		}
		torch.save(model_dict, results_path / experiment_name / file_name)
		Path(dataset_path).unlink()
		del dataset
		del model
		del model_dict
		del loader
