import networkx as nx
import numpy as np
from rdkit import Chem
import torch
import torch_geometric as pyg
from tqdm import tqdm
from typing import Optional
from scipy import sparse

from ..data.mol import MorganGenerator, SortAndSlice
from .base import BaseGraph, BaseGraphTokenizer


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
    bond_types : dict
        Mapping from RDKit bond types to integers.
    morgan : MorganGenerator
        MorganGenerator object to generate hashed identifiers.
    """


    bond_types = {
        Chem.rdchem.BondType.SINGLE: 0,
        Chem.rdchem.BondType.DOUBLE: 1,
        Chem.rdchem.BondType.AROMATIC: 2,
        Chem.rdchem.BondType.TRIPLE: 3,
        }

    def __init__(
        self,
        verbose: bool = False,
        morgan_kwargs: dict = {},
        **kwargs
    ):
        super(MorganGraph, self).__init__(verbose=verbose)
        self.morgan = MorganGenerator(**morgan_kwargs)

    def make_graph(self, mol: Chem.Mol):
        """
        Convert a molecule into a graph.

        Parameters
        ----------
        mol : Chem.Mol
            RDKit molecule object.
        
        Returns
        -------
        pyg.data.Data
            PyTorch Geometric Data object.
            x : torch.Tensor
                Node features. Shape (n, m), where n is the number of atoms in mol
                and m is maximum radii of the Morgan identifiers.
            edge_index : torch.Tensor
                Edge index tensor. Shape (2, 2e), where e is the number of bonds in mol.
                For each bond between atoms i and j, there are two edges (i, j) and (j, i).
            edge_attr : torch.Tensor
                Edge attribute tensor. Shape (2e, 1).
                Contains the bond type for each edge as integers; single (0), double (1),
                aromatic (2), triple (3).
        """
        envs = self.morgan.environments(mol)
        x = torch.tensor(envs, dtype=torch.long)
        num_bonds = mol.GetNumBonds()
        edge_index = torch.full(
            (2, num_bonds*2), fill_value=-1, dtype=torch.long
        )
        edge_attr = torch.full(
            (num_bonds*2, 1), fill_value=-1, dtype=torch.int
        )

        for idx, bond in enumerate(mol.GetBonds()):
            start = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            edge_index[0, idx] = start
            edge_index[1, idx] = end
            edge_index[0, idx + num_bonds] = end
            edge_index[1, idx + num_bonds] = start
            bond_type = self.bond_types[bond.GetBondType()]
            edge_attr[idx, 0] = bond_type
            edge_attr[idx + num_bonds, 0] = bond_type

        return pyg.data.Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr
        )
            
class MorganGraphTokenizer(BaseGraphTokenizer):
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
    def __init__(
        self,
        X: list[Chem.Mol],
        y: np.ndarray = None,
        train: np.ndarray = np.array([]),
        test: np.ndarray = np.array([]),
        transform_kwargs: dict = {},
        verbose: bool = False,
    ):
        super(MorganGraphTokenizer, self).__init__(
            X=X, y=y, train=train, test=test,
            transform_kwargs=transform_kwargs, verbose=verbose
        )

    def _transform_base(self, **kwargs):
        """
        Base transformation method as MorganGraph.
        """
        return MorganGraph(verbose=self.verbose, **kwargs)

class SNSGraphTokenizer(MorganGraphTokenizer):
    """
    Graph tokenizer using Morgan hashed identifiers with sort and slice.

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
        Keyword arguments for the MorganGraph and SortAndSlice.
    verbose : bool
        Whether to print progress information.

    Attributes
    ----------
    vocab_size : int
        Number of unique hashed identifiers. Excludes unknown token.
    envs : list[np.ndarray]
        List of hashed identifiers arrays for each molecule.
    sort_and_slice : SortAndSlice
        SortAndSlice object to sort and slice hashed identifiers.
    """
    def __init__(
        self,
        X: list[Chem.Mol],
        y: np.ndarray = None,
        train: np.ndarray = np.array([]),
        test: np.ndarray = np.array([]),
        transform_kwargs: dict = {},
        verbose: bool = False,
    ):
        super(SNSGraphTokenizer, self).__init__(
            X=X, y=y, train=train, test=test,
            transform_kwargs=transform_kwargs, verbose=verbose
        )
        if 'vocab_size' in transform_kwargs:
            self.vocab_size = transform_kwargs.pop('vocab_size')
        else:
            self.vocab_size = 1000

        self.envs = [np.array(graph.x, dtype=int) if graph != None else None for graph in self.X.values()]
        self.sort_and_slice = SortAndSlice(
            generator=self.transform.morgan, fpsize=self.vocab_size, verbose=False  
        )
        self.sort_and_slice.verbose = self.verbose
        self.reset(self.train_idx, self.test_idx)

    def encode(self, idx: int) -> None:
        """
        Encode hashed identifiers into sort and slice integer ranks.
        For molecular graph at index idx in self.X, self.X.x is
        updated with the encoded identifiers. Unknown identifiers
        are encoded as the maximum rank.

        Parameters
        ----------
        idx : int
            Index of the molecule in the dataset.
        """
        atomic_environments = self.envs[idx]
        if atomic_environments is None:
            return
        x = np.full(atomic_environments.shape, fill_value=np.nan)
        encoder = self.sort_and_slice.encoder
        unk = encoder['unk']
        for i in range(atomic_environments.shape[0]):
            for j in range(atomic_environments.shape[1]):
                env = atomic_environments[i, j]
                x[i, j] = encoder.get(env, unk)
        self.X[idx].x = torch.tensor(x, dtype=torch.long)

    def reset(self, train: np.ndarray, test: np.ndarray, vocab_size: None) -> None:
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
        self.sort_and_slice.clear()
        self.sort_and_slice.update(train_environments)
        self.sort_and_slice.sort()
        self.sort_and_slice.slice(vocab_size)
        self.sort_and_slice.encoder['unk'] = len(self.sort_and_slice.encoder)
        for i in self.X:
            self.encode(i)  
        self.vocab_size = len(self.sort_and_slice.encoder)
        if self.verbose:
            print(f'Vocabulary size: {self.vocab_size} (including unknown token).')

"""
    TODO: Implement MolETokenizer
"""

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