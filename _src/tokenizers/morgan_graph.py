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
