from _src.nn import get_nn
from _src.tokenizers import get_tokenizer
from _src.models import get_model
from _src.data.utils import load_dataset
from _src.data.datasets import BaseDataFrame
from _src.data.mol import MorganGenerator, SortAndSlice
from _src.data.datasets import GraphDataset

from _src.nn.pred_head import (BinaryHead, RegressionHead, MultiClassHead, MultiTaskLoss)

from pathlib import Path
import numpy as np
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
    learning_rate: float = config['learning_rate']
    weight_decay: float = config.get('weight_decay', 0.0)
    targets: dict = config['targets']
    splits: list[str] = config.get('splits', [])
    neptune_run = config.get('neptune_run')
    
    model_class = get_nn(config['model'])
    model_kwargs = config.get('model_kwargs', {})
    print(f'\n##################################################\n') if verbose else None
    print(f'Pretraining run {name}') if verbose else None
    print(f'Tokenizer: {tokenizer_class}') if verbose else None
    print(f'Model: {config['model']}') if verbose else None
    print(f'Data: {pretrain_data}') if verbose else None
    print(f'Splits: {splits}') if verbose else None
    print(f'Batch size: {batch_size}') if verbose else None
    print(f'Epochs: {epochs}') if verbose else None
    print(f'Learning rate: {learning_rate}') if verbose else None
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


    # prepare raw graphs
    GraphDataset(
        root=root,
        tokenizer=tokenizer,
        molecules=df.rdkit_mols,
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

        pretrain_loader = pyg.loader.DataLoader(
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
        model['main'].to(device)
        graph_0 = pretrain_dataset[0].to(device)
        print(f'Graph 0: {graph_0}') if verbose else None
        out = model['main'](graph_0.x, graph_0.edge_index, graph_0.edge_attr)
        head_input_dim = out['global_state'].size(-1)
        targets_key = pretrain_dataset.targets
        
        is_regression = torch.full((len(targets_key),), False)
        head_kwargs = {}
        
        for i, target_name in enumerate(targets_key):
            head_name = targets_key[target_name]['prediction_head']
            head_cls = pred_head_map[head_name]
            
            output_dim = graph_0[target_name].size(1)

            if head_name == 'multiclass' or head_name == 'binary':
                class_weights = targets_key[target_name]['class_weights'].to(device)
            
            else:
                class_weights = None
                is_regression[i] = True

            head_kwargs[target_name] = {
                'input_dim': head_input_dim,
                'hidden_dim': 256,
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

        model = model.to(device)
        model.train()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        for epoch in range(epochs):
            
            epoch_loss = 0
            epoch_scores = torch.zeros(len(targets_key), device=device)
            pbar = tqdm(total=len(pretrain_loader), desc=f'Epoch {epoch+1}/{epochs} | Batch Loss: {torch.nan}', disable=not verbose,)
            for batch_num, batch in enumerate(pretrain_loader):
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model['main'](x=batch.x, edge_index=batch.edge_index, batch=batch.batch)
                losses = torch.empty(len(targets_key), device=device)
                scores = torch.empty(len(targets_key), device=device)
                for i, target_name in enumerate(targets_key):
                    head = model[target_name]
                    if targets_key[target_name]['level'] == 'global':
                        embed = out['global_state']
                        embed.to(device)
                        y = batch[target_name].type(embed.dtype)
                        losses[i] = head.loss(x=embed, y=y,)
                        scores[i] = head.score(x=embed, y=y,)
                    elif targets_key[target_name]['level'] == 'node':
                        embed = out['final_state']
                        y = batch[target_name].type(embed.dtype)
                        mask = batch[f'{target_name}_mask'].type(embed.dtype)
                        losses[i] = head.loss(x=embed, y=y, mask=mask,)
                        score_now = head.score(x=embed, y=y, mask=mask,)
                    scores[i] = score_now
                    if neptune_run is not None:
                        neptune_run[f'{split}/batch_{target_name}_score'].append(score_now.item())
                    epoch_scores[i] += scores[i].item()


                loss = model['losses'](losses)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                if neptune_run is not None:
                    neptune_run[f'{split}/batch_loss'].append(loss.item())
                
                pbar.set_description(f'Epoch {epoch+1}/{epochs} | Batch Loss: {loss.item():.4f}')
                pbar.update(1)
                
            average_loss = epoch_loss / len(pretrain_loader)
            pbar.set_description(f'Epoch {epoch+1}/{epochs} | Epoch Loss: {average_loss:.4f} | Last Batch Loss: {loss.item():.4f}')
            pbar.close()
            if neptune_run is not None:
                average_scores = epoch_scores / len(pretrain_loader)
                neptune_run[f'{split}/epoch_average_loss'].append(average_loss)
                for i, target_name in enumerate(targets_key):
                    score_now = average_scores[i].item()
                    neptune_run[f'{split}/{target_name}_epoch_average_score'].append(score_now)
                     
        model_dict = {
            'tokenizer': tokenizer.to_dict(),
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
            head_kwargs = head_kwargs[target_name]

            model_dict['heads'][target_name] = {
                'state': state,
                'cls': head_cls,
                'kwargs': head_kwargs,
            }
        
        torch.save(model_dict, results_path / experiment_name / file_name)
        Path(pretrain_dataset.processed_paths[0]).unlink()
        del pretrain_dataset
        del model
        del model_dict
        del pretrain_loader
       