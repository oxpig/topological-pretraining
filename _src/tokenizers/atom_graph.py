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
     
    def __init__(
        self,
        molecules: list[Chem.Mol] = None,
        atom_types: dict = {
            'C' : 0,
            'O' : 1,
            'N' : 2,
        },
        bond_types = {
            Chem.rdchem.BondType.SINGLE: 0,
            Chem.rdchem.BondType.DOUBLE: 1,
            Chem.rdchem.BondType.TRIPLE: 2,
            Chem.rdchem.BondType.AROMATIC: 3,
        },
        verbose: bool = False
    ):
        super(AtomGraph, self).__init__(verbose)

        if molecules is not None:
            atom_types = {}
            bond_types = {}
            for mol in molecules:
                if mol is None:
                    continue
                for atom in mol.GetAtoms():
                    atom = atom.GetSymbol()
                    if atom not in atom_types:
                        atom_types[atom] = len(atom_types)

                for bond in mol.GetBonds():
                    bond = bond.GetBondType()
                    if bond not in bond_types:
                        bond_types[bond] = len(bond_types)

        self.atom_types = atom_types
        self.bond_types = bond_types

    def make_graph(self, mol: Chem.Mol):
        x = torch.empty((mol.GetNumAtoms(), 1), dtype=torch.int)
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            atom_type = self.atom_types.get(symbol, len(self.atom_types))
            x[atom.GetIdx()] = atom_type

        num_bonds = mol.GetNumBonds()

        edge_index = torch.full(
            (2, num_bonds*2), fill_value=-1, dtype=torch.long
        )
        edge_attr = torch.full(
            (num_bonds*2, 1), fill_value=-1, dtype=torch.int
        )

        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_type = self.bond_types[bond.GetBondType()]

            edge_index[0, bond.GetIdx()] = start
            edge_index[1, bond.GetIdx()] = end
            edge_index[0, bond.GetIdx() + num_bonds] = end
            edge_index[1, bond.GetIdx() + num_bonds] = start
            edge_attr[bond.GetIdx()] = bond_type
            edge_attr[bond.GetIdx() + num_bonds] = bond_type

        return pyg.data.Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )


class AtomGraphTokenizer(BaseTokenizer):
     
    def __init__(
        self,
        X: list[Chem.Mol],
        y: np.ndarray = None,
        train: np.ndarray = np.array([]),
        test: np.ndarray = np.array([]),
        transform_kwargs: dict = {},
        verbose: bool = False,
    ):
        if len(train) == 0:
            train = np.arange(len(X)) 
        transform_kwargs['molecules'] = [X[i] for i in train]
        super(AtomGraphTokenizer, self).__init__(
            X=X, y=y, train=train, test=test,
            transform_kwargs=transform_kwargs, verbose=verbose
        )

    @property
    def train(self):
        return [self.X[i] for i in self.train_idx], self.y[self.train_idx]
    
    @property
    def test(self):
        return [self.X[i] for i in self.test_idx], self.y[self.test_idx]

    def _transform_base(self, **kwargs):
        return AtomGraph(verbose=self.verbose, **kwargs)
    
    def reset(self, train: np.ndarray, test: np.ndarray) -> None:
        self.train_idx = train
        self.test_idx = test
        self.transform_kwargs['molecules'] = [self.origin_X[i] for i in train]
        self.set_transform(self.transform_kwargs)
        self.X = self.transform(self.origin_X)

    