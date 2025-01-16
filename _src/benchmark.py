from .utils import get_model, get_tokenizer
from .data.utils import load_dataset
from .data.datasets import BaseDataset

import numpy as np
from pathlib import Path

def benchmark(config):

    data_path = config['data']
    verbose: bool = config['verbose']
    benchmark_data = config['benchmark']
    model_kwargs = config.get('model_kwargs', {})
    model_class = get_model(config['model'])
    
    tokenizer_class = get_tokenizer(config['tokenizers'])
    tokenizer_kwargs = config.get('tokenizer_kwargs', {})

    for benchmark in benchmark_data:
        print(f'Benchmarking on {benchmark}') if verbose else None
        df: BaseDataset = load_dataset(
            name=benchmark, root=data_path, compression=True,
            verbose=verbose,
        )
        splits = df.splits
        mols = df.rdkit_mols
        y = df.y.to_numpy()
        tokenizer = tokenizer_class(X=mols, y=y, **tokenizer_kwargs)

        out = np.zeros((len(splits.columns), len(df)))

        for idx, split in enumerate(splits.columns):
            print(f'Processing split {split}') if verbose else None
            split = splits.loc[:,split]
            train = split[split == 'Train'].index
            test = split[split == 'Test'].index
            tokenizer.reset(train, test)
            train_X, train_y = tokenizer.train
            
            model = model_class(task=df.task, **model_kwargs)
            model.fit(train_X, train_y)
            train_pred = model.predict(train_X)
            out[idx, train] = train_pred
            test_X, _ = tokenizer.test
            test_pred = model.predict(test_X)
            out[idx, test] = test_pred
        out_path = Path(data_path) / 'predictions' / config['tokenizers']
        out_path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path / f'{benchmark}.npz', out)

