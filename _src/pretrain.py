from .utils import get_model, get_tokenizer, get_nn
from .data.utils import load_dataset
from .data.datasets import BaseDataFrame
from .data.mol import MorganGenerator, SortAndSlice
from .data.datasets import GraphDataset

from .nn.pred_head import (BinaryHead, RegressionHead, MultiClassHead)

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
        for target_name in pretrain_dataset.targets:
            head_name = pretrain_dataset.targets[target_name]['prediction_head']
            head_cls = pred_head_map[head_name]
            level = pretrain_dataset.targets[target_name]['level']
            output_dim = graph_0[target_name].size(1)

            if head_name == 'multiclass' or head_name == 'binary':
                class_weights = pretrain_dataset.targets[target_name].get('class_weights', None)

            head = head_cls(
                input_dim=head_input_dim,
                hidden_dim=512,
                output_dim=output_dim,
                num_layers=2,
                dropout=0.0,
                batch_norm=True,
                act='hardswish',
                class_weights=None
            )

            model[target_name] = head

        model.eval()
        out = model['main'](
            pretrain_dataset[0].x, pretrain_dataset[0].edge_index, batch=pretrain_dataset[0].batch
        )
        model['head'] = BinaryHead(
            input_dim=out['global_state'].size(1), hidden_dim=512, num_layers=2, output_dim=sns.fpsize, act='hardswish', batch_norm=True
        )
        model.to(device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            for batch in pretrain_loader:
                batch.to(device) 
                optimizer.zero_grad()
                out = model['main'](batch.x, batch.edge_index, batch=batch.batch)
                loss = model['head'].loss(out['global_state'], batch.y)
                loss.backward()
                optimizer.step()
            break

       