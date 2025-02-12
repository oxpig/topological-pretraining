import torch
import torch_geometric as pyg

import numpy as np
from rdkit import Chem

from .base import BaseGraph, BaseTokenizer

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
    molecules : list[Chem.Mol]|None
        List of RDKit molecule objects.
    atom_types : dict
        Mapping from atom symbols to integers.
        Defaults to {'C': 0, 'O': 1, 'N': 2, 'unk': 3} if no molecules are provided.
    bond_types : dict
        Mapping from RDKit bond types to integers.
        Defaults to {'SINGLE': 0, 'DOUBLE': 1, 'AROMATIC': 2, 'TRIPLE': 3}.

    Examples
    --------
    >>> from rdkit import Chem
    >>> mol = Chem.MolFromSmiles('CC(=O)O')  # acetic acid
    >>> atom_graph = AtomGraph()
    >>> data = atom_graph.make_graph(mol)
    >>> print(data.x)  # atom types
    tensor([[0],
            [0],
            [1],
            [1]], dtype=torch.int32)
    >>> print(data.edge_index)  # connectivity
    tensor([[0, 1, 1, 1, 2, 3],
            [1, 2, 3, 0, 1, 1]])
    >>> print(data.edge_attr)  # bond types
    tensor([[0],
            [1],
            [0],
            [0],
            [1],
            [0]], dtype=torch.int32)
    ------------------------------------------------------------------------
    >>> from rdkit import Chem
    >>> molecules = ['CC(=O)O', 'CCO', 'FCCN', 'C#CCl']
    >>> mols = [Chem.MolFromSmiles(i) for i in molecules]
    >>> atom_graph = AtomGraph(molecules=mols)
    >>> data = atom_graph.make_graph(mols[-1])
    >>> print(data.x)  # atom types
    tensor([[0],
            [0],
            [4]], dtype=torch.int32)
    >>> print(data.edge_index)  # connectivity
    tensor([[0, 1, 1, 2],
            [1, 2, 0, 1]])
    >>> print(data.edge_attr)  # bond types
    tensor([[0],
            [0],
            [0],
            [0]], dtype=torch.int32)
    """
    atom_types = {}
    bond_types = {
            Chem.rdchem.BondType.SINGLE: 0,
            Chem.rdchem.BondType.DOUBLE: 1,
            Chem.rdchem.BondType.AROMATIC: 2,
            Chem.rdchem.BondType.TRIPLE: 3,
        }

    def reset(self, mols: list[Chem.Mol]) -> None:
        atom_types = {}
        for m in mols:
            if m is None:
                continue
            for atom in m.GetAtoms():
                atom = atom.GetSymbol()
                if atom not in atom_types:
                    atom_types[atom] = len(atom_types)

        # add unknown atom type
        atom_types['UNK'] = len(atom_types)
        self.atom_types = atom_types

    def make_graph(self, mol: Chem.Mol):
        """
        Convert a molecule into a graph.

        Parameters
        ----------
        mol : Chem.Mol
            RDKit molecule object.

        Returns
        -------
        torch_geometric.data.Data
            PyTorch Geometric Data object.
            x : torch.Tensor
                Node features. Shape (n, 1), where n is the number of atoms in mol.
            edge_index : torch.Tensor
                Edge indices. Shape (2, 2*m), where m is the number of bonds in mol.
            edge_attr : torch.Tensor
                Edge attributes. Shape (2*m, 1).
        """
        # initialize node features
        x = torch.empty((mol.GetNumAtoms(), 1), dtype=torch.int)
        # set unknown atom type
        unk = self.atom_types['UNK']

        # loop over atoms and set atom types
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            atom_type = self.atom_types.get(symbol, unk)
            x[atom.GetIdx()] = atom_type

        # initialize edge index and edge attributes
        num_bonds = mol.GetNumBonds()
        edge_index = torch.full(
            (2, num_bonds*2), fill_value=-1, dtype=torch.long
        )
        edge_attr = torch.full(
            (num_bonds*2, 1), fill_value=-1, dtype=torch.int
        )

        # loop over bonds and set edge index and edge attributes
        for bond in mol.GetBonds():
            # get start and end atom indices
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_type = self.bond_types[bond.GetBondType()]

            # set edge index
            edge_index[0, bond.GetIdx()] = start
            edge_index[1, bond.GetIdx()] = end

            # set reverse edge index
            edge_index[0, bond.GetIdx() + num_bonds] = end
            edge_index[1, bond.GetIdx() + num_bonds] = start
            
            # set bond types
            edge_attr[bond.GetIdx()] = bond_type
            edge_attr[bond.GetIdx() + num_bonds] = bond_type

        return pyg.data.Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )


class AtomGraphTokenizer(BaseTokenizer):
    """
    Class to tokenize molecules into graphs using atom types and bond types.

    Parameters
    ----------
    transform_kwargs : dict
        Keyword arguments for the AtomGraph.
    verbose : bool
        Whether to print progress information.
    """
    def __init__(
        self,
        transform_kwargs: dict = {},
        verbose: bool = False,
        **kwargs,
    ):
        super(AtomGraphTokenizer, self).__init__(
            transform_kwargs=transform_kwargs, verbose=verbose
        )
        if 'atom_types' in transform_kwargs:
            self.transform.atom_types = transform_kwargs['atom_types']
        if 'bond_types' in transform_kwargs:
            self.transform.bond_types = transform_kwargs['bond_types']

    def _transform_base(self, **kwargs):
        return AtomGraph(verbose=self.verbose, **kwargs)
    
    def reset(self, mols: Chem.Mol) -> None:
        self.transform.reset(mols)

    @property
    def vocab_size(self):
        return len(self.transform.atom_types)
    
    @property
    def bond_types(self):
        return self.transform.bond_types

    @property
    def atom_types(self):
        return self.transform.atom_types
    
    def to_dict(self):
        params = super().to_dict()
        params['atom_types'] = self.atom_types
        params['bond_types'] = self.bond_types
        return params