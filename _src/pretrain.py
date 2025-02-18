from .utils import get_model, get_tokenizer, get_nn
from .data.utils import load_dataset
from .data.datasets import BaseDataFrame
from .data.mol import MorganGenerator, SortAndSlice
from .data.datasets import GraphDataset

from .nn.pred_head import (BinaryHead, RegressionHead, MultiClassHead, MultiTaskLoss)

from pathlib import Path
import numpy as np
import torch
import torch_geometric as pyg

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
    data_path: str = config['data']
    results_path: str = config['results']
    verbose: bool = config['verbose']
    pretrain_data: list[str]|str = config['pretrain_data']
    tokenizer_class = config['tokenizer']
    tokenizer_kwargs = config.get('transform_kwargs', {})
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size: int = config['batch_size']
    epochs: int = config['epochs']
    learning_rate: float = config['learning_rate']
    targets: dict = config['targets']
    
    model_class = get_nn(config['model'])
    model_kwargs = config.get('model_kwargs', {})
    print(f'\n##################################################\n') if verbose else None
    print(f'Pretraining run {name}') if verbose else None
    print(f'Tokenizer: {tokenizer_class}') if verbose else None
    print(f'Model: {config['model']}') if verbose else None
    print(f'Data: {pretrain_data}') if verbose else None
    # print(f'Target: {target}') if verbose else None
    print(f'Batch size: {batch_size}') if verbose else None
    print(f'Epochs: {epochs}') if verbose else None
    print(f'Learning rate: {learning_rate}') if verbose else None
    print(f'Device: {device}') if verbose else None
    print(f'\n##################################################\n') if verbose else None

    # Load the dataset as a dataframe
    df: BaseDataFrame = load_dataset(name=pretrain_data, root=data_path, verbose=verbose)
    molecules = df.rdkit_mols

    # Load the tokenizer
    tokenizer = get_tokenizer(tokenizer_class, tokenizer_kwargs)

    root = Path(data_path) / pretrain_data / tokenizer_class
    root.mkdir(parents=True, exist_ok=True)
    raw_dir = root / 'raw'

    if not raw_dir.exists():
        raw_dir.mkdir()
        for i, mol in enumerate(molecules):
            graph = tokenizer.raw(mol)
            graph.idx = i
            torch.save(graph, raw_dir / f'{i}.pt')

    splits = [col for col in df.columns if 'butina_filter' in col]
    if len(splits) == 0:
        print('No splits found, using all data for pretraining.') if verbose else None
        df['butina_filter'] = 1
        splits = ['butina_filter']
    # gen = MorganGenerator(verbose=verbose, radius=2, chirality=True)
    # loss = torch.nn.BCELoss()

    # prepare raw graphs
    GraphDataset(
        root=root,
        tokenizer=tokenizer,
        molecules=df.rdkit_mols,
        fit_tokenizer=False,
    )

    for split in splits:
        
        idx = df[split]
        pretrain_dataset = GraphDataset(
            root=root, split=(split, idx), tokenizer=tokenizer,
            targets=targets, run_id=name, fit_tokenizer=True,
        )


        pretrain_loader = pyg.data.DataLoader(pretrain_dataset, batch_size=batch_size, shuffle=True)
        model = torch.nn.ModuleDict({
            'main': model_class(**model_kwargs),
        })
        graph_0 = pretrain_dataset[0]
        out = model['main'](graph_0.x, graph_0.edge_index, graph.edge_attr)
        head_input_dim = out['global_state'].size(-1)
        targets_key = pretrain_dataset.targets
        is_regression = torch.full((len(targets_key),), False)
        
        for i, target_name in enumerate(targets):
            head_name = targets[target_name]['prediction_head']
            head_cls = pred_head_map[head_name]
            
            output_dim = graph_0[target_name].size(1)

            if head_name == 'multiclass' or head_name == 'binary':
                class_weights = targets[target_name]['class_weights']
            
            else:
                class_weights = None
                is_regression[i] = True

            head = head_cls(
                input_dim=head_input_dim,
                hidden_dim=512,
                output_dim=output_dim,
                num_layers=2,
                dropout=0.0,
                batch_norm=True,
                act='hardswish',
                class_weights=class_weights
            )
            model[target_name] = head

        if len(is_regression) == 1:
            model['losses'] = torch.nn.Identity()

        else:
            model['losses'] = MultiTaskLoss(is_regression=is_regression)


        model.to(device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            average_epoch_loss = 0
            for batch_num, batch in enumerate(pretrain_loader):
                batch.to(device) 
                optimizer.zero_grad()
                out = model['main'](batch.x, batch.edge_index, batch=batch.batch)
                losses = torch.empty(len(targets_key), device=device)
                for i, target_name in enumerate(targets_key):
                    head = model[target_name]
                    if targets_key['level'] == 'global':
                        losses[i] = head.loss(
                            x = out['global_state'],
                            y = batch[target_name],
                        )
                    elif targets_key['level'] == 'node':
                        losses[i] = head.loss(
                            x = out['final_state'],
                            y = batch[target_name],
                            mask = batch[f'{target_name}_mask']
                        )
                loss = model['losses'](losses)
                loss.backward()
                optimizer.step()
                average_epoch_loss += loss.item() / batch_num
            
        model['tokenizer'] = pretrain_dataset.tokenizer
        torch.save(model, results_path / name / f'model_{split}.pt')
       