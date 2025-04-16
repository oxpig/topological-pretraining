import torch
import torch_geometric as pyg

import numpy as np
from rdkit import Chem

from .base import BaseGraph, GraphTokenizer

import torch
import torch_geometric as pyg
from tqdm import tqdm

import numpy as np
from rdkit import Chem


class AtomGraph(BaseGraph):
    """
    Class to convert a molecule into a graph using atom types and bond types.

    Parameters
    ----------
    atom_types : dict, optional
        Mapping from atomic numbers to integers.
        Will be automatically generated if not provided.
    bond_types : dict, optional
        Mapping from RDKit bond types to integers.
        Defaults to {SINGLE: 0, DOUBLE: 1, AROMATIC: 2, TRIPLE: 3}.
    **kwargs : dict
        Additional arguments passed to BaseGraph.

    Examples
    --------
    >>> from rdkit import Chem
    >>> mol = Chem.MolFromSmiles('CC(=O)O')  # acetic acid
    >>> graph = AtomGraph()
    >>> graph.reset([mol])  # initialize atom types
    >>> data = graph.raw(mol)  # create raw graph
    >>> encoded = graph.encode(data)  # encode atom types
    >>> print(encoded.x)  # encoded atom types
    tensor([[0],
            [0],
            [1],
            [1]], dtype=torch.long)
    >>> print(encoded.edge_index)  # connectivity
    tensor([[0, 1, 1, 2],
            [1, 0, 2, 1]])
    >>> print(encoded.edge_attr)  # bond types
    tensor([[0],
            [0],
            [1],
            [1]], dtype=torch.int)
    """
    
    def __init__(self, node_types: dict = None, edge_types: dict = None, **kwargs):
        super(AtomGraph, self).__init__(node_types=node_types, edge_types=edge_types)

    def get_nodes(self, mol: Chem.Mol):
        """
        Get the raw node descriptor for an atom.
        """
        x = torch.full((mol.GetNumAtoms(), 1), fill_value=-1, dtype=torch.long)
        for atom in mol.GetAtoms():
            x[atom.GetIdx()] = atom.GetAtomicNum()
        return x
    
    @property
    def x_size(self):
        """
        Get the size of the node descriptor. 
        (i.e. the number of tokens / features per node in the raw graph)
        """
        return 1
    
    @property
    def edge_attr_size(self):
        """
        Get the size of the edge descriptor. 
        (i.e. the number of tokens / features per edge in the raw graph)
        """
        return 1

class AtomGraphTokenizer(GraphTokenizer):

    def _transform_base(self, **kwargs):
        return AtomGraph(verbose=self.verbose, **kwargs)