from __future__ import annotations
from _src.tokenizers import load_tokenizer
from _src.tokenizers.targets import Targets

import numpy as np
from pathlib import Path
import torch_geometric as pyg
from torch_geometric.data.dataset import _repr
import torch
import copy
from tqdm import tqdm
from rdkit import Chem


from typing import Optional, Callable, TYPE_CHECKING
import warnings

if TYPE_CHECKING:
    from _src.tokenizers import GraphTokenizer
    

class PreFilter:
    """
    Class for filtering dataset based on split indicies

    Parameters:
    ----------
    split : tuple[str, torch.Tensor])
        Tuple with a split name at 0 and split indices at 1.
    """
    def __init__(self, split: tuple[str, torch.Tensor] = None):
        if split:
            self.split_name, self.indices = split
        else:
            self.split_name, self.indices = None, None

    def _pre_filter(self, data: pyg.data.Data):
        """
        Check if graph is in split.

        Parameters:
        ----------
        data : torch_geometric.data.Data
            A PyTorch Geometric Data object. Must have an `idx` attribute 
            (int) that indicates the object's position in the dataset.

        Returns:
        -------
        bool
            True if `data.idx` is in the split, or if no split is provided.
            Otherwise False.

        Raises:
        ------
        AttributeError
            If `data` does not have an `idx` attribute.
        TypeError
            If `data.idx` is not an integer.
        """
        if self.indices is None:
            return True
        else:
            if "idx" not in data:
                raise AttributeError(
                    "`data`, requires an integer `idx` attribute " \
                    "indicating the objects position in the dataset."
                )
            if not isinstance(data.idx, int):
                raise TypeError(
                    "`data` attribute `idx` must be an integer."
                )
            return self.indices[data.idx]
        
    def __call__(self, data: pyg.data.Data):
        return self._pre_filter(data)

class GraphDataset(pyg.data.InMemoryDataset):
    """
    Graph dataset for pretraining.

    Parameters:
    ----------
    root : str
        Path to store or retrieve molecular dataset.
    tokenizer : GraphTokenizer
        Tokenizer that converts molecules into PyTorch Geometric Data objects.
        See `_src/tokenizers.py` for definition
    molecules : Optional[List[rdkit.Chem.Mol]]
        A list of RDKit molecules.
    split : Optional[tuple[str, torch.Tensor]]
        A tuple containing a name for the split for saving, and a tensor of indices.
    fit_tokenizer : bool
        Boolean for whether to fit the GraphTokenizer.
        Defaults to True unless the input tokenizer has already been fitted.
    run_id : Optional[str]
        Optional id name for save paths.
    targets : dict[str, dict[str, str]] | None
        Nested dictionary of self-supervised target labels to generate.
        Keys are target names; can be `ECFP`, `SNS`, `PDV`, or `FCFP`
        Values are dictionaries of target variables, such as `radius` and `fpsize`.
        REVISIT
    verbose : bool
        Verbosity. Default is `False`.
    """
    _indexes = None
    def __init__(
        self, root: str, tokenizer: GraphTokenizer = None,
        molecules: Optional[list[Chem.Mol]] = None,
        split: Optional[tuple[str, torch.Tensor]] = None,
        fit_tokenizer: bool = True,
        run_id: Optional[str] = None,
        targets: dict[str, dict[str, str]] | None = None,
        verbose: bool = False,
    ):
        Path(root).mkdir(parents=True, exist_ok=True)
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
        
        if Path(self.processed_paths[0]).exists():
            print('Loading processed graphs into memory...') if self.verbose else None
            self.load(self.processed_paths[0])
            if self.targets is not None:          
                if not self.targets.is_fitted_:
                    print('Targets not fitted. Fitting...') if self.verbose else None
                    data_list = [graph for graph in self]
                    self.fit_targets(data_list)
            
    def get(self, idx: int):
        """
        Retrieve graph object at index `idx`.
        Altered version of the native method in PyTorch Geometric that
        computes target values on the fly. Preserves memory efficiency 
        for large target labels like ECFP fingerprints.

        Parameters:
        ----------
        idx : int
            Index of the graph object.
        
        Returns:
        -------
        torch_geometric.data.Data
            A PyTorch graph object.
        """
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
        if self.targets is not None:
            if self.targets.is_fitted_:
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
        molecules = [self.molecules[data.idx] for data in data_list]
        self.targets.fit((molecules, data_list))
        if self.targets_path.exists():
            warnings.warn('Overwriting existing targets.')
        self.targets.save(self.targets_path)

    def process(self):
        raw_dir = Path(self.raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
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
            with tqdm(total=len(molecules), desc='Processing graphs', disable=not self.verbose) as pbar:
                for idx, mol in enumerate(molecules):
                    raw_graph = self.tokenizer.raw(mol)
                    raw_graph.idx = idx
                    data_list.append(raw_graph.to_dict())
                    pbar.update(1)
            torch.save(data_list, raw_graph_path)
            print(f'Saved {len(data_list)} raw graphs to {raw_graph_path}.') if self.verbose else None
        
        processed_graph_path = Path(self.processed_paths[0])
        print(f'Processed graphs path: {processed_graph_path}.') if self.verbose else None
        if raw_graph_path.exists() and self.fit_tokenizer:
            data_list = torch.load(raw_graph_path, weights_only=False)
            data_list = [pyg.data.Data(**data_dict) for data_dict in data_list]
            print(f'Loaded {len(data_list)} raw graphs from {raw_graph_path}.') if self.verbose else None
            data_list = [graph for graph in data_list if self.pre_filter(graph)]
            if not self.tokenizer.is_fitted_ or self.fit_tokenizer:
                print("Fitting tokenizer...")
                self.tokenizer.fit(data_list)
                self.tokenizer.save(self.tokenizer_path, params_only=True)

            data_list = [self.tokenizer.encode(graph) for graph in data_list]
            self.save(data_list, processed_graph_path)
            if self.targets is not None:
                if not self.targets.is_fitted_:
                    print('Fitting targets...') if self.verbose else None
                    self.fit_targets(data_list)
            del data_list
        
    @property
    def raw_file_names(self):
        return [f'{i}.pt' for i in range(len(self))]

    def get_raw(self, idx):
        return torch.load(self.raw_paths[idx], weights_only=False)
    
    def __getitem__(self, idx):
        graph = super(GraphDataset, self).__getitem__(idx)
        if isinstance(graph, GraphDataset):
            for G in graph:
                if not isinstance(G, pyg.data.Data):
                    raise TypeError('Graph must be a PyG Data object.')
                if 'x' not in G:
                    raise Warning(f'Graph {int(G.idx)} does not contain node features.')
            return graph
        elif isinstance(graph, pyg.data.Data):
            if 'x' not in graph:
                raise Warning(f'Graph {int(graph.idx)} does not contain node features.')
            return graph
        else:
            raise ValueError('Graph must be a PyG Data object or subset of GraphDataset.')
    
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