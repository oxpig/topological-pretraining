from _src.nn import get_nn
from _src.nn.autoencoder import AutoEncoder
from _src.tokenizers import get_tokenizer
from _src.models import get_model
from _src.data.utils import load_dataset
from _src.data.datasets import BaseDataFrame
from _src.data.mol import MorganGenerator, SortAndSlice
from _src.data.datasets import GraphDataset
from _src.data.loader import DataLoader

from _src.nn.pred_head import (BinaryHead, RegressionHead, MultiClassHead, MultiTaskLoss)

from copy import deepcopy
from pathlib import Path
import numpy as np
from rdkit import Chem
import torch
import torch_geometric as pyg
from tqdm import tqdm




pred_head_map = {
    'binary': BinaryHead,
    'regression': RegressionHead,
    'multiclass': MultiClassHead,
}

# TODO: change to handle multiple targets

def pretrain(config: dict):
    """
    Run the pretraining process.
    """
    name: str = config['name']
    experiment_name: str = config['experiment']
    raw_name: str = config.get('raw_name', experiment_name)
    data_path: str = config['data']
    results_path: str = config['results']
    verbose: bool = config['verbose']
    pretrain_data: list[str]|str = config['pretrain_data']
    tokenizer_class = config['tokenizer']
    tokenizer_kwargs = config.get('tokenizer_kwargs', {})
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size: int = config['batch_size']
    epochs: int = config['epochs']
    # learning_rate: float = config['learning_rate']
    warmup_epochs: int = config.get('warmup_epochs', 0)
    lr_decay_half_life: int = config.get('lr_decay_half_life', 5)
    weight_decay: float = config.get('weight_decay', 0.0)
    targets: dict = config['targets']
    splits: list[str] = config.get('splits', [])
    neptune_run = config.get('neptune_run')
    seed = config['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    model_class = get_nn(config['model'])
    model_kwargs = config.get('model_kwargs', {})
    if model_kwargs.get('graph_pool_type', None) == 'global_node':
        tokenizer_kwargs['global_token'] = True
    else:
        tokenizer_kwargs['global_token'] = False

    print(f'\n##################################################\n') if verbose else None
    print(f'Pretraining run {name}') if verbose else None
    print(f'Tokenizer: {tokenizer_class}') if verbose else None
    print(f'Model: {config['model']}') if verbose else None
    print(f'Data: {pretrain_data}') if verbose else None
    print(f'Splits: {splits}') if verbose else None
    print(f'Batch size: {batch_size}') if verbose else None
    print(f'Epochs: {epochs}') if verbose else None
    print(f'Warmup epochs: {warmup_epochs}') if verbose else None
    print(f'Learning rate decay half life: {lr_decay_half_life}') if verbose else None
    print(f'Weight decay: {weight_decay}') if verbose else None
    print(f'Device: {device}') if verbose else None
    print(f'\n##################################################\n') if verbose else None

    # Load the dataset as a dataframe
    df: BaseDataFrame = load_dataset(name=pretrain_data, root=data_path, verbose=verbose)
    
    # Load the tokenizer
    tokenizer = get_tokenizer(tokenizer_class)(transform_kwargs=tokenizer_kwargs)

    root = Path(data_path) / pretrain_data / raw_name
    root.mkdir(parents=True, exist_ok=True)

    if len(splits) == 0:
        print('No splits found, using all data for pretraining.') if verbose else None
        df['butina_filter'] = 1
        splits = ['butina_filter']

    if not config.get('standardization', True):
        print('Loading molecules without standardization...') if verbose else None
        mols = df.SMILES.to_list()
        mols = [Chem.MolFromSmiles(mol) for mol in mols]
    
    else:
        mols = df.rdkit_mols

    # prepare raw graphs
    GraphDataset(
        root=root,
        tokenizer=tokenizer,
        molecules=mols,
        fit_tokenizer=False,
        verbose=verbose,
    )
    save_path = results_path / experiment_name
    save_path.mkdir(parents=True, exist_ok=True)
    print(f'Looping through splits: {splits}') if verbose else None
    for split in splits:
        assert split in df.columns, f'{split} not found in dataframe.'
        if len(splits) > 1:
            file_name = f'{name}_{split}.pt'
        else:
            file_name = f'{name}.pt'
        if (save_path / file_name).exists():
            print(f'Model {file_name} already exists, skipping.') if verbose else None
            continue
        idx = df[split]
        pretrain_dataset = GraphDataset(
            root=root, split=(split, idx), tokenizer=tokenizer,
            targets=targets, run_id=name, fit_tokenizer=True,
            verbose=verbose
        )

        pretrain_loader = DataLoader(
            pretrain_dataset, batch_size=batch_size, shuffle=True
        )
        model_kwargs['input_dim'] = pretrain_dataset[0].x.size(1)
        if 'node_embedding' in model_kwargs and isinstance(model_kwargs['node_embedding'], int):
            model_kwargs['node_embedding'] = (
                len(pretrain_dataset.tokenizer.node_types), model_kwargs['node_embedding']
            )

        model = torch.nn.ModuleDict({
            'main': model_class(device=device, **model_kwargs),
        })
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
        
        is_regression = torch.full((len(targets_key),), False)
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
        if neptune_run:
            neptune_run['model/num_params'] = num_params
            neptune_run['model/num_heads'] = len(targets_key)
            neptune_run['model/summary'] = str(model)
        model = model.to(device)
        model.train()
        lr = num_params ** -0.5
        if neptune_run:
            neptune_run[f'{split}/lr'] = lr
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
                    optimizer, start_factor=0.5, end_factor=1.0,
                    total_iters=warmup_steps
                ),
                torch.optim.lr_scheduler.ExponentialLR(
                    optimizer, gamma=gamma_for_halflife_5_epochs
                )
            ]
        )
        
        for epoch in range(epochs):
            
            epoch_loss = 0
            epoch_scores = torch.zeros(len(targets_key), device=device)
            pbar = tqdm(total=len(pretrain_loader), desc=f'Epoch {epoch+1}/{epochs} | Batch Loss: {torch.nan}', disable=not verbose,)
            for batch_num, batch in enumerate(pretrain_loader):
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model['main'](
                    x=batch.x, edge_index=batch.edge_index, batch=batch.batch,
                    global_idx=batch.get('global_idx'),
                )
                losses = torch.empty(len(targets_key), device=device)
                scores = torch.empty(len(targets_key), device=device)
                last_score = torch.tensor(0.0, device=device)
                for i, target_name in enumerate(targets_key):
                    head = model[target_name]
                    if targets_key[target_name]['level'] == 'global':
                        embed = out['global_state']
                        embed.to(device)
                        pred = head(embed)
                        y = batch[target_name].type(embed.dtype)
                        losses[i] = head.loss(pred=pred, y=y,)
                        if batch_num % 100 == 0:
                            score_now = head.score(pred=pred, y=y,)
                            last_score = deepcopy(score_now)
                        else:
                            score_now = None
                    elif targets_key[target_name]['level'] == 'node':
                        embed = out['final_state']
                        embed.to(device)
                        pred = head(embed)
                        y = batch[target_name].type(embed.dtype)
                        mask = batch[f'{target_name}_mask'].type(embed.dtype)
                        losses[i] = head.loss(pred=pred, y=y, mask=mask,)
                        if batch_num % 100 == 0:
                            score_now = head.score(pred=pred, y=y,)
                            last_score = deepcopy(score_now)
                        else:
                            score_now = None
                    else:
                        raise ValueError('Invalid target level.')
                    if score_now is not None:
                        if neptune_run is not None:
                            neptune_run[f'{split}/batch_{target_name}_score'].append(score_now.item())
                        scores[i] = score_now.item()
                        epoch_scores[i] += score_now.item()

                loss = model['losses'](losses)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                if neptune_run is not None:
                    neptune_run[f'{split}/batch_loss'].append(loss.item())
                
                pbar.set_description(f'Epoch {epoch+1}/{epochs} | Batch Loss: {loss.item():.4f} | Batch Score: {last_score.item():.4f}')
                pbar.update(1)
                scheduler.step()

            average_loss = epoch_loss / len(pretrain_loader)
            average_scores = epoch_scores / len(pretrain_loader)
            pbar.set_description(f'Epoch {epoch+1}/{epochs} | Epoch Loss: {average_loss:.4f} | Last Batch Loss: {loss.item():.4f} | Last Batch Score: {last_score.item():.4f}')
            pbar.close()
            if neptune_run is not None:
                neptune_run[f'{split}/epoch_average_loss'].append(average_loss)
                for i, target_name in enumerate(targets_key):
                    score_now = average_scores[i].item()
                    neptune_run[f'{split}/{target_name}_epoch_average_score'].append(score_now)

            if (epoch + 1) % 15 == 0:
                model_dict = {
                    'tokenizer': pretrain_dataset.tokenizer.to_dict(),
                    'main': {
                        'state': model['main'].state_dict(),
                        'cls': model_class.__name__,
                        'kwargs': model_kwargs
                    }
                }
                model_dict['heads'] = {}
                for i, target_name in enumerate(targets_key):
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
                
                torch.save(model_dict, results_path / experiment_name / f'epoch_{epoch+1}_{file_name}')

        model_dict = {
            'tokenizer': pretrain_dataset.tokenizer.to_dict(),
            'main': {
                'state': model['main'].state_dict(),
                'cls': model_class.__name__,
                'kwargs': model_kwargs
            }
        }
        model_dict['heads'] = {}
        for i, target_name in enumerate(targets_key):
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
    """
    name: str = config['name']
    experiment_name: str = config['experiment']
    raw_name: str = config.get('raw_name', experiment_name)
    data_path: str = config['data']
    results_path: str = config['results']
    verbose: bool = config['verbose']
    pretrain_data: list[str]|str = config['pretrain_data']
    tokenizer_class = config['tokenizer']
    tokenizer_kwargs = config.get('tokenizer_kwargs', {})
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size: int = config['batch_size']
    epochs: int = config['epochs']
    # learning_rate: float = config['learning_rate']
    warmup_epochs: int = config.get('warmup_epochs', 0)
    lr_decay_half_life: int = config.get('lr_decay_half_life', 5)
    weight_decay: float = config.get('weight_decay', 0.0)

    splits: list[str] = config.get('splits', [])
    neptune_run = config.get('neptune_run')
    seed = config['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    model_class = AutoEncoder
    model_kwargs = config.get('model_kwargs', {})

    print(f'\n##################################################\n') if verbose else None
    print(f'Pretraining run {name}') if verbose else None
    print(f'Tokenizer: {tokenizer_class}') if verbose else None
    print(f'Model: {config['model']}') if verbose else None
    print(f'Data: {pretrain_data}') if verbose else None
    print(f'Splits: {splits}') if verbose else None
    print(f'Batch size: {batch_size}') if verbose else None
    print(f'Epochs: {epochs}') if verbose else None
    print(f'Warmup epochs: {warmup_epochs}') if verbose else None
    print(f'Learning rate decay half life: {lr_decay_half_life}') if verbose else None
    print(f'Weight decay: {weight_decay}') if verbose else None
    print(f'Device: {device}') if verbose else None
    print(f'\n##################################################\n') if verbose else None

    # Load the dataset as a dataframe
    df: BaseDataFrame = load_dataset(name=pretrain_data, root=data_path, verbose=verbose)
    
    # Load the tokenizer
    tokenizer = get_tokenizer(tokenizer_class)(transform_kwargs=tokenizer_kwargs)

    root = Path(data_path) / pretrain_data / raw_name
    root.mkdir(parents=True, exist_ok=True)

    if len(splits) == 0:
        print('No splits found, using all data for pretraining.') if verbose else None
        df['butina_filter'] = 1
        splits = ['butina_filter']

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
        assert split in df.columns, f'{split} not found in dataframe.'
        if len(splits) > 1:
            file_name = f'{name}_{split}.pt'
        else:
            file_name = f'{name}.pt'
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
            tokenizer.fit(split_mols)
            dataset = tokenizer.transform(split_mols)
            if isinstance(dataset, np.ndarray):
                dataset = torch.tensor(dataset)
            torch.save(dataset, dataset_path)

        if model_kwargs.get('class_weights') is not None:
            if model_kwargs['decoder_type'] == 'multiclass':
                dataset_summed = dataset.sum(dim=0, dtype=torch.float64)
                dataset_summed[dataset_summed == 0] = dataset_summed[dataset_summed == 0] + 1e-8
                weights = torch.tensor(dataset.size(0) / (dataset.size(1) * dataset_summed))
                weights.type(torch.float64)
            elif model_kwargs['decoder_type'] == 'binary':
                zero_class = torch.zeros_like(dataset, dtype=torch.float32)
                zero_class[dataset == 0] = 1
                zero_class_sum = zero_class.sum(dim=0, dtype=torch.float32)
                zero_class_sum[zero_class_sum == 0] = zero_class_sum[zero_class_sum == 0] + 1
                data_sum = dataset.sum(dim=0, dtype=torch.float32)
                data_sum[data_sum == 0] = data_sum[data_sum == 0] + 1
                weights = torch.stack([
                    dataset.size(0) / (2 * zero_class_sum),
                    dataset.size(0) / (2 * data_sum)
                ])
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
        if neptune_run:
            neptune_run['model/num_params'] = num_params
            neptune_run['model/summary'] = str(model)
        model = model.to(device)
        model.train()
        lr = num_params ** -0.5
        if neptune_run:
            neptune_run[f'{split}/lr'] = lr
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
                    optimizer, start_factor=0.5, end_factor=1.0,
                    total_iters=warmup_steps
                ),
                torch.optim.lr_scheduler.ExponentialLR(
                    optimizer, gamma=gamma_for_halflife_5_epochs
                )
            ]
        )
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_scores = []
            last_score = np.nan
            pbar = tqdm(total=len(loader), desc=f'Epoch {epoch+1}/{epochs} | Batch Loss: {torch.nan}', disable=not verbose,)
            for batch_num, batch in enumerate(loader):
                batch = batch[0]
                batch = batch.to(dtype=torch.float32, device=device)
                optimizer.zero_grad()
                pred = model.pred(x=batch)
                pred_vals = pred.detach().cpu().numpy()
                pred_vals = np.where(pred_vals > 0.5, 1, 0)
                sum_vals = pred_vals.sum()
                sum_x = batch.sum()
                overlap = np.logical_and(pred_vals, batch.cpu().numpy()).sum() / np.prod(batch.size())
                if batch_num % 100 == 0:
                    score = model.decoder.score(y=batch, pred=pred)
                    if neptune_run is not None:
                        neptune_run[f'{split}/batch_score'].append(score.item())
                    last_score = score.item()
                loss = model.decoder.loss(y=batch, pred=pred)
                loss.backward()
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()
                if neptune_run is not None:
                    neptune_run[f'{split}/batch_loss'].append(loss.item())
                pbar.set_description(f'Epoch {epoch+1}/{epochs} | Batch Loss: {loss.item():.4f} | Batch Score: {last_score:.4f} | overlap: {overlap:.4f}')
                pbar.update(1)
            average_loss = epoch_loss / len(loader)
            average_score = sum(epoch_scores) / len(epoch_scores)
            pbar.set_description(f'Epoch {epoch+1}/{epochs} | Epoch Loss: {average_loss:.4f} | Epoch Score: {average_score:.4f}')
            pbar.close()

            if neptune_run is not None:
                neptune_run[f'{split}/epoch_average_loss'].append(average_loss)
                neptune_run[f'{split}/epoch_average_score'].append(average_score)
            
            if (epoch + 1) % 15 == 0:
                model_dict = {
                    'tokenizer': tokenizer.to_dict(),
                    'state': model.state_dict(),
                    'cls': model_class.__name__,
                    'kwargs': model_kwargs
                }
                torch.save(model_dict, results_path / experiment_name / f'epoch_{epoch+1}_{file_name}')
            
        model_dict = {
            'tokenizer': tokenizer.to_dict(),
            'state': model.state_dict(),
            'cls': model_class.__name__,
            'kwargs': model_kwargs
        }
        torch.save(model_dict, results_path / experiment_name / file_name)
        Path(dataset_path).unlink()
        del dataset
        del model
        del model_dict
        del loader
        

            
