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
    Works for datasets upto ~1,000,000 samples with 32Gb of memory.

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
        Retrieve molecular graph object at index `idx`.
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
            A molecule as a PyTorch graph object.
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
                data = self.compute_graph_targets(data)
                self._data_list[idx] = copy.copy(data)

        return data
    
    def compute_graph_targets(self, graph: pyg.data.Data):
        """
        Compute the target labels for molecule and add to graph object.

        Parameters:
        ----------
        graph : torch_geometric.data.Data
            A molecule as a PyTorch graph object.
        
        Returns:
        -------
        torch_geometric.data.Data
            The input graph with target labels as attributes.
        """
        idx = graph.idx.item()
        # Retrieve RDKit molecule for graph
        mol = self.molecules[idx] 
        graph = self.targets.transform(mol, graph)
        return graph
    
    @property
    def tokenizer_path(self):
        """
        Path to save and retrieve graph tokenizer.

        Returns:
        -------
        pathlib.Path
            Path to saved GraphTokenizer.
        """
        return Path(self.processed_dir) / f'tokenizer{self.run_id}{self.split_name}.pt'
    
    @property
    def targets_path(self):
        """
        Path to save and retrieve graph target generator.

        Returns:
        pathlib.Path
            Path to saved Target.
        """
        return Path(self.processed_dir) / f'targets{self.run_id}{self.split_name}.pt'

    @property
    def processed_file_names(self):
        """
        Path for processed files.
        Only one file for saving all graphs.

        Returns:
        -------
        list[str]
            Path to processed graphs in a List.
        """
        return [f'processed{self.run_id}{self.split_name}.pt']
    
    def fit_targets(self, data_list: list[pyg.data.Data]):
        """
        Fit the target object to a list of molecular graphs.
        Target object gets save to self.targets_path
        This is for instances where the target depends on a set of
        data, e.g., Sort and Slice fingerprints.

        Parameters:
        ----------
        data_list : list[pyg.data.Data]
            List of molecules as PyTorch graph objects.

        Returns:
        -------
        None
        """
        print('Fitting targets...') if self.verbose else None
        molecules = [self.molecules[data.idx] for data in data_list]
        self.targets.fit((molecules, data_list))
        if self.targets_path.exists():
            warnings.warn('Overwriting existing targets.')
        self.targets.save(self.targets_path)

    def process(self):
        """
        Process raw molecular data.
        Makes raw graphs if they do not exist.
        Filters graphs using split indices.
        Fits GraphTokenizer to filtered raw graphs.
        Saves GraphTokenizer as a dictionary.
        Tokenizes filtered raw graphs and saves to processed path.
        Fits Targets to tokenized graphs
        """
        self.make_raw_dir()
        raw_graphs_path = self.raw_graphs_path
        self.save_molecules(self._molecules)
        self.make_raw_graphs()
        data_list = self.load_raw_graphs()
        
        processed_graph_path = Path(self.processed_paths[0])
        print(f'Processed graphs path: {processed_graph_path}.') if self.verbose else None
        if raw_graphs_path.exists() and self.fit_tokenizer:
            
            print(f'Loaded {len(data_list)} raw graphs from {raw_graphs_path}.') if self.verbose else None
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

    def make_raw_dir(self):
        """
        Make raw directory.
        """
        raw_dir = Path(self.raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)

    @property
    def raw_file_names(self):
        """
        File names for raw graphs and RDKit molecules.

        Returns:
        -------
        List[str]
            List of file names. 0th elements is the file name for raw graphs.
            1st element is the file name for RDKit molecules.
        """
        return ['graphs.pt', 'molecules.pt']

    @property
    def molecules_path(self):
        return Path(self.raw_paths[1])

    @property
    def raw_graphs_path(self):
        return Path(self.raw_paths[0])
    
    @property
    def processed_graphs_path(self):
        return Path(self.processed_paths[0])
    
    def check_raw_graphs(self):
        """
        Method to check raw graphs have been made.

        Returns:
        -------
        bool
            `False` if saved raw graphs do not exist
            `True` if saved raw graphs do exist

        Raises:
        ------
        FileNotFoundError
            If no paths for raw graphs or molecules exist.
            If path for raw graphs does not exist, molecules are needed to create graphs.
        """
        if not self.raw_graphs_path.exists() and not self.molecules_path.exists():
            # if no files raise an Error
            raise FileNotFoundError('No molecules or graphs found in the raw directory.')
        elif not self.raw_graphs_path.exists() and self.molecules_path.exists():
            # If raw graphs do not exist return False
            return False
        else:
            # If raw graphs exist
            return True
        
    def check_processed_graphs(self):
        """
        Check if processed graphs exist on disk.

        Returns:
        -------
        bool
            `True` if processed graphs exist otherwise `False`.
        """
        return self.processed_graphs_path.exists()
        
    def make_raw_graphs(self):
        """
        Method for making raw graphs and saving to disk.
        E.g., with an AtomGraphTokenizer raw node features are atomic numbers.

        Raw graphs are saved as dictionaries.
        """
        if self.check_raw_graphs(): return
        molecules = self.molecules
        data_list = []
        with tqdm(total=len(self.molecules), desc='Making raw graphs', disable=not self.verbose) as pbar:
            for idx in range(len(molecules)):
                raw_graph = self.get_raw(idx)
                data_list.append(raw_graph.to_dict())
                pbar.update(1)
        torch.save(data_list, self.raw_graphs_path)
        print(f'Saved {len(data_list)} raw graphs to {self.raw_graphs_path}.') if self.verbose else None

    def load_raw_graphs(self):
        """
        Load all raw graphs from file.

        Returns:
        -------
        list[torch_geometric.data.Data] 
            List of molecules as raw graphs.
        """
        graphs = torch.load(self.raw_graphs_path, weights_only=False)
        graphs = [pyg.data.Data(**data_dict) for data_dict in graphs]
        return graphs

    def get_raw(self, idx: int):
        """
        Compute the raw graph of a molecule on the fly.

        Parameters:
        ----------
        idx : int
            The index of a graph in the dataset.

        Returns:
        -------
        torch_geometric.data.Data
            A molecule as a graph data object.
        """
        molecule = self.molecules[idx]
        graph = self.tokenizer.raw(molecule)
        graph.idx = idx
        return graph
    
    def __getitem__(self, idx: int):
        """
        Retrieve a processed molecular graph.

        Parameters:
        ----------
        idx : int
            The index of a graph in the dataset.

        Returns:
        -------
        torch_geometric.data.Data
            A molecule as a graph data object.
        """
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
    
    def save_molecules(self, molecules: list[Chem.Mol | None] = None):
        """
        Save a list of molecules to disk.
        Skips saving if path to molecules already exists.

        Parameters:
        ----------
        molecules : list[rdkit.Chem.Mol | None]
            A list of rdkit molecules to save.
            Accepts `None` in list to preserve index positions in raw graph list of molecules that
            fail RDKit parsing.

        Raises:
        ------
        TypeError
            If `molecules` is not a list and a saved version of molecules does not exist.
        TypeError
            If any type other than rdkit molecules or None appears in the input list.
        """
        if self.molecules_path.exists():
            warnings.warn(
                'Path to molecules already exists. Skipping save.',
                UserWarning
            )
        if not isinstance(molecules, list): 
            raise TypeError(f'Expect `list`. Got: {type(molecules)}')
        if not all(isinstance(m, Chem.Mol|None) for m in molecules):
            raise TypeError('Found type other than `rdkit.Chem.Mol` and `None` in molecules')
        torch.save(molecules, self.molecules_path)
        print(f'Saved {len(molecules)} molecules to {self.molecules_path}.') if self.verbose else None

    @property
    def molecules(self):
        """
        Load the original RDKit molecules.

        Returns:
        -------
        List[rdkit.Chem.Mol | None]

        Raises:
        ------
        ValueError
            If no molecules are found in memory or on disk.
        """
        if self._molecules is not None:
            return self._molecules
        elif self.molecules_path.exists():
            self._molecules = torch.load(self.molecules_path, weights_only=False)
            return self._molecules
        else:
            raise ValueError('Molecules not found.')
            
    
    @property
    def priors(self):
        return {target: self.targets[target]['prior'] for target in self.targets}