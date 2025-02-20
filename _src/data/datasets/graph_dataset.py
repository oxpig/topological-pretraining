from ...tokenizers import GraphTokenizer, load_tokenizer
from ...tokenizers.targets import Targets

import numpy as np
from pathlib import Path
import torch_geometric as pyg
from torch_geometric.data.dataset import _repr
import torch
import copy
from tqdm import tqdm
from rdkit import Chem

from typing import Optional, Callable
import warnings


class PreFilter:

    def __init__(self, split: tuple[str, torch.Tensor]):
        if split:
            self.split_name, self.indices = split
        else:
            self.split_name, self.indices = None, None

    def _pre_filter(self, data: pyg.data.Data):
        
        if self.indices is None:
            return True
        else:
            return self.indices[data.idx]
        
    def __call__(self, data: pyg.data.Data):
        return self._pre_filter(data)

class GraphDataset(pyg.data.InMemoryDataset):
    """
    Graph dataset for pretraining.
    """
    _indexes = None
    def __init__(
        self, root: str, tokenizer: GraphTokenizer = None,
        molecules: Optional[list[Chem.Mol]] = None,
        split: Optional[tuple[str, torch.Tensor]] = None,
        fit_tokenizer: bool = True,
        run_id: Optional[str] = None,
        targets: dict[str, dict[str, str]] = None,
        verbose: bool = False,
    ):
        if run_id is None:
            run_id = ''
        else:
            run_id = f'_{run_id}'
        if split is not None:
            split = (f'_{split[0]}', split[1])
        else:
            split = ('', None)

        self.split_name, self.split_indices = split
        self._molecules = molecules
        self.run_id = run_id
        self.verbose = verbose
        
        
        tokenizer_path = Path(root) / 'processed' / f'tokenizer{run_id}{self.split_name}.pt'
        if tokenizer_path.exists():
            tokenizer = load_tokenizer(tokenizer_path)
            self.fit_tokenizer = False

        if tokenizer is None:
            raise ValueError('Tokenizer not found.')
            
        self.tokenizer = tokenizer
        self.fit_tokenizer = fit_tokenizer
        
        targets_path = Path(root) / 'processed' / f'targets{run_id}{self.split_name}.pt'
        if targets_path.exists():
            print('Loading targets...') if self.verbose else None
            self.targets = Targets(targets_path=targets_path)
        elif isinstance(targets, dict):
            print('Creating targets...') if self.verbose else None
            self.targets = Targets(targets=targets.copy())
        else:
            self.targets = None

        super(GraphDataset, self).__init__(
            root=root, pre_filter=PreFilter(split),
        )
        if self.targets is not None:          
            if not self.targets.is_fitted_:
                print('Targets not fitted. Fitting...') if self.verbose else None
                data_list = [self.get(i) for i in range(len(self))]
                self.fit_targets(data_list)

        print('Loading processed graphs into memory...') if self.verbose else None
        if Path(self.processed_paths[0]).exists():
            self.load(self.processed_paths[0])

    def get(self, idx: int):
        if self.len() == 1:
            return copy.copy(self._data)

        if not hasattr(self, '_data_list') or self._data_list is None:
            self._data_list = self.len() * [None]
        elif self._data_list[idx] is not None:
            return copy.copy(self._data_list[idx])

        data = pyg.data.separate.separate(
            cls=self._data.__class__,
            batch=self._data,
            idx=idx,
            slice_dict=self.slices,
            decrement=False,
        )
        data = self.load_graph_targets(data)

        self._data_list[idx] = copy.copy(data)

        return data

    def load_graph_targets(self, graph: pyg.data.Data):
        idx = graph.idx.item()
        mol = self.molecules[idx]
        graph = self.targets.transform(mol, graph)
        return graph
    
    @property
    def tokenizer_path(self):
        return Path(self.processed_dir) / f'tokenizer{self.run_id}{self.split_name}.pt'
    
    @property
    def targets_path(self):
        return Path(self.processed_dir) / f'targets{self.run_id}{self.split_name}.pt'

    @property
    def processed_file_names(self):
        return [f'processed{self.run_id}{self.split_name}.pt']
    
    def fit_targets(self, data_list: list[pyg.data.Data]):
        print('Fitting targets...') if self.verbose else None
        self.targets.fit((self.molecules, data_list))
        if self.targets_path.exists():
            warnings.warn('Overwriting existing targets.')
        self.targets.save(self.targets_path)

    def process(self):
        raw_dir = Path(self.raw_dir)
        molecules_path = raw_dir / 'molecules.pt'
        raw_graph_path = raw_dir / 'graphs.pt'
        if self._molecules is not None and not molecules_path.exists():
            assert all(isinstance(m, Chem.Mol|None) for m in self._molecules)
            torch.save(self._molecules, molecules_path)
            print(f'Saved {len(self._molecules)} molecules to {molecules_path}.') if self.verbose else None

        if not raw_graph_path.exists() and not molecules_path.exists():
            raise FileNotFoundError('No molecules or graphs found in the raw directory.')
        if not raw_graph_path.exists() and molecules_path.exists():
            molecules = torch.load(molecules_path, weights_only=False)
            data_list = []
            for idx, mol in enumerate(molecules):
                raw_graph = self.tokenizer.raw(mol)
                raw_graph.idx = idx
                data_list.append(raw_graph.to_dict())
            torch.save(data_list, raw_graph_path)
            print(f'Saved {len(data_list)} raw graphs to {raw_graph_path}.') if self.verbose else None
        
        processed_graph_path = Path(self.processed_paths[0])

        if raw_graph_path.exists() and self.fit_tokenizer:
            data_list = torch.load(raw_graph_path, weights_only=False)
            data_list = [pyg.data.Data(**data_dict) for data_dict in data_list]
            print(f'Loaded {len(data_list)} raw graphs from {raw_graph_path}.') if self.verbose else None
            data_list = [graph for graph in data_list if self.pre_filter(graph)]
            if not self.tokenizer.is_fitted_:
                self.tokenizer.fit(data_list)
                self.tokenizer.save(self.tokenizer_path, params_only=True)

            data_list = [self.tokenizer.encode(graph) for graph in data_list]
            self.save(data_list, processed_graph_path)
            if self.targets is not None:
                if not self.targets.is_fitted_:
                    self.fit_targets(data_list)
        
        else:
            return
        
    @property
    def raw_file_names(self):
        return [f'{i}.pt' for i in range(len(self))]

    def get_raw(self, idx):
        return torch.load(self.raw_paths[idx], weights_only=False)
    
    def __getitem__(self, idx):
        graph = super(GraphDataset, self).__getitem__(idx)
        if 'x' not in graph:
            raise Warning('Graph does not contain node features.')
        return graph
    
    @property
    def molecules(self):
        molecules_path = Path(self.raw_dir) / 'molecules.pt'
        if self._molecules is not None:
            return self._molecules
        elif molecules_path.exists():
            self._molecules = torch.load(molecules_path, weights_only=False)
            return self._molecules
        else:
            raise ValueError('Molecules not found.')
    
    @property
    def priors(self):
        return {target: self.targets[target]['prior'] for target in self.targets}