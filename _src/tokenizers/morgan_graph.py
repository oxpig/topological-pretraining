import networkx as nx
import numpy as np
from rdkit import Chem
import torch
import torch_geometric as pyg
from tqdm import tqdm
from typing import Optional
from scipy import sparse

from ..data.mol import MorganGenerator, SortAndSlice
from .base import BaseGraph, GraphTokenizer


class MorganGraph(BaseGraph):
    """
    Class to convert a molecule into a graph using Morgan hashed identifiers.

    Parameters
    ----------
    verbose : bool
        Whether to print progress information.
    morgan_kwargs : dict
        Keyword arguments for the MorganGenerator.

    Attributes
    ----------
    env_types : dict
        Mapping from hashed identifiers to integer tokens.
    bond_types : dict
        Mapping from RDKit bond types to integers.
    morgan : MorganGenerator
        MorganGenerator object to generate hashed identifiers.
    """
    def __init__(
        self,
        verbose: bool = False,
        node_types: dict = {},
        edge_types: dict = {},
        max_vocab_size: int = 2048,
        global_token: bool = False,
        **kwargs
    ):
        super(MorganGraph, self).__init__(
            node_types=node_types, edge_types=edge_types, max_vocab_size=max_vocab_size,
            verbose=verbose, global_token=global_token
        )
        morgan = MorganGenerator(**kwargs)
        self.sort_and_slice = SortAndSlice(
            generator=morgan, verbose=verbose  
        )
    
    def get_nodes(self, mol):
        envs = self.sort_and_slice.generator.environments(mol)
        x = torch.tensor(envs, dtype=torch.long)
        return x
    
    def reset(self, mols: list[Chem.Mol|pyg.data.Data]) -> None:
        """
        Reset the hashed identifiers for a new set of molecules.

        Parameters
        ----------
        mols : list[Chem.Mol]
            List of RDKit molecule objects.
        """
        self.sort_and_slice.clear()
        if all(isinstance(m, pyg.data.Data) for m in mols):
            print(mols[0])
            mols = pyg.data.Batch.from_data_list(mols)
            assert torch.all(mols.raw), 'Data must be raw graphs.'
            envs = mols.x.numpy()
            self.sort_and_slice.append(envs)
        elif all(isinstance(m, Chem.Mol|None) for m in mols):
            self.sort_and_slice.update(mols)
        else:
            raise ValueError('Molecules must be RDKit molecule objects or PyG Data objects.')
    
        self.sort_and_slice.sort()
        self.sort_and_slice.slice(self.max_vocab_size)
        self.sort_and_slice.encoder['UNK'] = len(self.sort_and_slice.encoder)

        if self.global_token:
            num_tokens_per_node = self.sort_and_slice.generator.radius + 1
            for i in range(num_tokens_per_node):
                self.sort_and_slice.encoder[f'GLOBAL_{i}'] = len(self.sort_and_slice.encoder)
        self.node_types = self.sort_and_slice.encoder



class MorganGraphTokenizer(GraphTokenizer):
    """
    Class to tokenize molecules into graphs using Morgan hashed identifiers.

    Parameters
    ----------
    X : list[Chem.Mol]
        List of RDKit molecule objects.
    y : Optional[np.ndarray]
        Array of labels.
    train : Optional[np.ndarray]
        Indices of the training set.
    test : Optional[np.ndarray]
        Indices of the test set.
    transform_kwargs : dict
        Keyword arguments for the MorganGraph.
    verbose : bool
        Whether to print progress information.
    """

    def _transform_base(self, **kwargs):
        """
        Base transformation method as MorganGraph.
        """
        return MorganGraph(verbose=self.verbose, **kwargs)


class MolETokenizer(MorganGraphTokenizer):
    """
    Currently not implemented.
    """
    def __init__(
        self, X: list[Chem.Mol], y: Optional[np.ndarray] = None,
        train: Optional[np.ndarray] = np.array([]), test: Optional[np.ndarray] = np.array([]),
        transform_kwargs = {}, verbose = False,
    ):
        morgan_kwargs = transform_kwargs.get('morgan_kwargs', {})
        morgan_kwargs['radius'] = 2
        transform_kwargs['morgan_kwargs'] = morgan_kwargs
        super(MorganGraphTokenizer, self).__init__(
            X=X, y=y, train=train, test=test,
            transform_kwargs=transform_kwargs, verbose=verbose
        )
        if 'vocab_size' in transform_kwargs:
            self.vocab_size = transform_kwargs.pop('vocab_size')
        else:
            self.vocab_size = 1000

        self.envs = [np.array(
            graph.x, dtype=int
            ) if graph != None else None for graph in self.X.values()
        ]
        self.sort_and_slice_atom = SortAndSlice(
            generator=self.transform.morgan, fpsize=self.vocab_size, verbose=False  
        )
        self.sort_and_slice_env = SortAndSlice(
            generator=self.transform.morgan, fpsize=self.vocab_size, verbose=False  
        )
        self.all_distance_matrices()
        self.reset(self.train_idx, self.test_idx)

    def encode(self, idx):
        atomic_environments = self.envs[idx]
        if atomic_environments is None:
            return
        
        x = np.full((atomic_environments.shape[0], 1), fill_value=np.nan)
        y = np.full((atomic_environments.shape[0], 1), fill_value=np.nan)
        encoder_0 = self.sort_and_slice_atom.encoder
        encoder_2 = self.sort_and_slice_env.encoder
        for i in range(atomic_environments.shape[0]):
            atom = atomic_environments[i, 0]
            env = atomic_environments[i, 2]
            x[i, 0] = encoder_0.get(atom, encoder_0['unk'])
            y[i, 0] = encoder_2.get(env, encoder_2['unk'])
        self.X[idx].x = torch.tensor(x, dtype=torch.long)
        self.X[idx].y = torch.tensor(y, dtype=torch.long)

    def reset(self, train: np.ndarray, test: np.ndarray, vocab_size: int = None) -> None:
        """
        Reset the training and test indices and re-encode the hashed identifiers.
        SortAndSlice object is updated with the new training environments.

        Parameters
        ----------
        train : np.ndarray
            Indices of the training set.
        test : np.ndarray
            Indices of the test set.
        """
        self.train_idx = train
        self.test_idx = test
        train_environments = [self.envs[i] for i in train]
        radius_0_envs = [env[:,0].reshape(-1, 1) for env in train_environments]
        self.sort_and_slice_atom.verbose = self.verbose
        self.sort_and_slice_atom.update(radius_0_envs)
        self.sort_and_slice_atom.sort()
        self.sort_and_slice_atom.slice(vocab_size)
        self.sort_and_slice_atom.encoder['unk'] = len(self.sort_and_slice_atom.encoder)
        self.sort_and_slice_atom.encoder['pad'] = len(self.sort_and_slice_atom.encoder)
        self.sort_and_slice_atom.encoder['mask'] = len(self.sort_and_slice_atom.encoder)
        
        radius_2_envs = [env[:,2].reshape(-1, 1) if env is not None else None for env in self.envs]
        self.sort_and_slice_env.verbose = self.verbose
        self.sort_and_slice_env.update(radius_2_envs)
        self.sort_and_slice_env.sort()
        self.sort_and_slice_env.slice(fpsize=len(self.sort_and_slice_env.identifiers))
        self.sort_and_slice_env.encoder['unk'] = len(self.sort_and_slice_env.encoder)

        for i in self.X:
            self.encode(i)
        self.vocab_size = len(self.sort_and_slice_atom.encoder)
        if self.verbose:
            print(f'Vocabulary size: {self.vocab_size} (including unknown, pad, and mask tokens).')

    def all_distance_matrices(self):
        pbar = tqdm(
            total=len(self.X), desc='Calculating distance matrices',
            disable=not self.verbose
        )
        for idx in self.X:
            self.distance_matrix(idx)
            pbar.update(1)
        pbar.close()

    def distance_matrix(self, idx):
        graph = self.X[idx]
        if graph is None:
            return None
        num_atoms = graph.x.shape[0]
        out = torch.full((num_atoms, num_atoms), fill_value=-1, dtype=torch.long)
        indexes = torch.full((2, num_atoms*num_atoms), fill_value=-1, dtype=torch.long)
        graph = pyg.utils.to_networkx(graph)
        count = 0
        for i in range(num_atoms):
            for j in range(i, num_atoms):
                distance = nx.shortest_path_length(graph, source=i, target=j)
                out[i, j] = distance
                
                indexes[0, count] = i
                indexes[1, count] = j
                count += 1
                if i == j:
                    continue
                out[j, i] = distance
                indexes[1, count] = j
                indexes[0, count] = i
                count += 1
        out = out.flatten().unsqueeze(1)

        self.X[idx].distances = out
        self.X[idx].dist_indexes = indexes
        


    def mask(self, idx):
        graph = self.X[idx]
        if graph is None:
            return None
        num_atoms = graph.x.shape[0]
        mask = torch.full((num_atoms, 1), fill_value=False, dtype=torch.bool)
        mask[graph.x == self.sort_and_slice_atom.encoder['pad']] = True
        self.X[idx].mask = mask

    def pad(self):
        return self.sort_and_slice_atom.encoder['pad']