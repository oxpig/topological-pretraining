from ... import tokenizers

from pathlib import Path
import torch_geometric as pyg
from torch_geometric.data.dataset import _repr
import torch
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
        self, root: str, tokenizer: tokenizers.GraphTokenizer = None,
        molecules: Optional[list[Chem.Mol]] = None,
        split: Optional[tuple[str, torch.Tensor]] = None,
    ):
        if split is not None:
            split = (f'_{split[0]}', split[1])
        else:
            split = ('', None)
        self.split = split[0]
        self.molecules = molecules
        if tokenizer is None:
            tokenizer_path = Path(root) / 'processed' / f'tokenizer{split[0]}.pt'
            if tokenizer_path.exists():
                tokenizer = tokenizers.load_tokenizer(tokenizer_path)
                
            else:
                raise ValueError('No tokenizer provided or found in the processed directory.')
        self.tokenizer = tokenizer
            
        super(GraphDataset, self).__init__(
            root=root, pre_filter=PreFilter(split)
        )
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f'processed{self.split}.pt']
    
    @property
    def tokenizer_path(self):
        return Path(self.processed_dir) / f'tokenizer{self.split}.pt'

    def process(self):

        raw_dir = Path(self.raw_dir)
        molecules_path = raw_dir / 'molecules.pt'
        raw_graph_path = raw_dir / 'graphs.pt'
        if self.molecules is not None:
            assert all(isinstance(m, Chem.Mol|None) for m in self.molecules)
            root = Path(self.root)
            raw = root / 'raw'
            raw.mkdir(parents=True, exist_ok=True)
            molecules_path = raw / 'molecules.pt'
            torch.save(self.molecules, raw / 'molecules.pt')

        processed_graph_path = Path(self.processed_paths[0])

        if processed_graph_path.exists():
            return
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
            print(f'Saved {len(data_list)} graphs to {raw_graph_path}.')
            
        if raw_graph_path.exists():
            data_list = pyg.io.fs.torch_load(raw_graph_path)
            data_list = [pyg.data.Data(**data_dict) for data_dict in data_list]
            print(f'Loaded {len(data_list)} graphs from {raw_graph_path}.')

            data_list = [graph for graph in data_list if self.pre_filter(graph)]
            if not self.tokenizer.fitted:
                self.tokenizer.fit(data_list)
                self.tokenizer.save(self.tokenizer_path, params_only=True)

            data_list = [self.tokenizer.encode(graph) for graph in data_list]
            self.save(data_list, processed_graph_path)

        else:
            return
        
    @property
    def raw_file_names(self):
        return [f'{i}.pt' for i in range(len(self))]
        
    def reset(self, indexes: torch.Tensor):
        self._indexes = indexes
        graphs = [self.get_raw(i) for i in self._indexes]
        self._tokenizer.fit(graphs)
        self.transform = self._tokenizer.preprocess

    def get_raw(self, idx):
        return torch.load(self.raw_paths[idx], weights_only=False)
    
    def __getitem__(self, idx):
        graph = super(GraphDataset, self).__getitem__(idx)
        if 'x' not in graph:
            raise Warning('Graph does not contain node features.')
        return graph

    

    
    