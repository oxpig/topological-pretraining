import torch
import torch_geometric as pyg

import numpy as np
from rdkit import Chem

from _src.tokenizers.base import BaseGraph, GraphTokenizer
from _src.data.encoder import OneHotEncoder

import torch
import torch_geometric as pyg
from tqdm import tqdm

import numpy as np
from rdkit import Chem

from typing import TYPE_CHECKING


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
    def empty_graph(self):
        return pyg.data.Data(
                raw=True,
                empty=True,
                x=torch.full((1, 1), fill_value=self.node_types["UNK"], dtype=torch.long),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                edge_attr=torch.empty((0, 1), dtype=torch.long),
            )
    
class AtomGraphTokenizer(GraphTokenizer):

    def _transform_base(self, **kwargs):
        return AtomGraph(verbose=self.verbose, **kwargs)

class AtomFeatureGraph(BaseGraph):
    """
    Based on atomic featurization by Dablander et al. 
    (https://jcheminf.biomedcentral.com/articles/10.1186/s13321-023-00708-w/figures/4)
    """
    def __init__(
        self, node_types: dict = None, edge_types: dict = None,
        max_vocab_size: int = None, verbose: bool = False, global_token: bool = False
    ):
        super(AtomFeatureGraph, self).__init__(
            node_types=node_types, edge_types=edge_types,
            max_vocab_size=max_vocab_size, verbose=verbose,
            global_token=global_token
        )
        self.atom_encoder = OneHotEncoder(
            categories=self.permitted_atoms
        )
        self.n_heavy_neighbors = OneHotEncoder(
            categories=[0, 1, 2, 3, 4,]
        )
        self.formal_charge = OneHotEncoder(
            categories=[-3, -2, -1, 0, 1, 2, 3,]
        )
        self.hybridization = OneHotEncoder(
            categories=["S", "SP", "SP2", "SP3", "SP3D", "SP3D2"]
        )
        self.chirality = OneHotEncoder(
            categories=[
                "CHI_UNSPECIFIED", 
                "CHI_TETRAHEDRAL_CW",
                "CHI_TETRAHEDRAL_CCW",
            ]
        )
        self.hydrogen_neighbors = OneHotEncoder(
            categories=[0, 1, 2, 3, 4,]
        )

    @property
    def n_features(self):
        mol = Chem.MolFromSmiles('C')
        atom = mol.GetAtomWithIdx(0)
        return self.get_atom_features(atom).shape[-1]

    def get_atom_features(self, atom: Chem.Atom):

        atom_type = self.atom_encoder.transform(atom.GetSymbol())
        n_heavy_neighbors = self.n_heavy_neighbors.transform(
            atom.GetDegree()
        )
        formal_charge = self.formal_charge.transform(atom.GetFormalCharge())
        hybridization = self.hybridization.transform(atom.GetHybridization())
        
        other_feats = np.array([[
            atom.IsInRing(), atom.GetIsAromatic(), # ring features
            float((atom.GetMass() - 10.812)/116.092), # scaled atomic mass
            float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5)/0.6), # scaled van der Waals radius
            float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64)/0.76), # scaled covalent radius
        ]], dtype=float)
        
        chirality = self.chirality.transform(atom.GetChiralTag())

        hydrogen_neighbors = self.hydrogen_neighbors.transform(
            atom.GetTotalNumHs()
        )
        return np.hstack([
            atom_type, n_heavy_neighbors, formal_charge,
            hybridization, other_feats, 
            chirality, hydrogen_neighbors,
        ]).astype(np.float32)

    def get_nodes(self, mol: Chem.Mol):
        """
        Get the raw node descriptor for an atom.
        """
        x = torch.full((mol.GetNumAtoms(), self.n_features), fill_value=-1, dtype=torch.float32)
        for atom in mol.GetAtoms():
            x[atom.GetIdx()] = torch.tensor(self.get_atom_features(atom))
        return x
        
    def encode(self, graph: pyg.data.Data):
        if not graph.raw:
            raise ValueError('Graph must be raw.')
        if 'x' not in graph:
            return graph
        graph = graph.clone()
        graph.raw = False
        if len(graph.x) > 0:
            graph = self.add_global_token(graph)

        return graph
        
       
    @property
    def permitted_atoms(self):
        return [
            'C','N','O','S','F','Si','P','Cl','Br',
            'Mg','Na','Ca','Fe','As','Al','I','B',
            'V','K','Tl','Yb','Sb','Sn','Ag','Pd',
            'Co','Se','Ti','Zn','Li','Ge','Cu','Au',
            'Ni','Cd','In','Mn','Zr','Cr','Pt','Hg',
            'Pb'
        ]

class AtomFeatureTokenizer(GraphTokenizer):
    """
    Tokenizer for AtomFeatureGraph.
    """
    def _transform_base(self, **kwargs):
        return AtomFeatureGraph(verbose=self.verbose, **kwargs)
    
    