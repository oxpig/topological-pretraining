import numpy as np
from rdkit import Chem
import torch
import torch_geometric as pyg
from tqdm import tqdm
from typing import Optional
from sklearn.base import BaseEstimator, TransformerMixin


class BaseTokenizer(BaseEstimator, TransformerMixin):
    """
    Base class for tokenizers.

    This class provides a framework for tokenizing molecules into a specific format.

    Parameters:
    ----------
    transform_kwargs : dict
        Keyword arguments for the tokenizer's transformation function.
    verbose : bool
        Whether to print progress information during tokenization.
    is_fitted_ : bool, optional
        Whether the tokenizer is already fitted. If None, it defaults to False.
        This is used for tokenizers that do not require fitting, such as ECFP fingerprints.
    """
    transform = lambda x: x
    fixed_transform_kwargs = {}
    precomputed = False
    is_fitted_ = False

    def __init__(
        self,
        transform_kwargs: dict = {},
        verbose: bool = False,
        is_fitted_: bool = None,
    ):
        super(BaseTokenizer, self).__init__()
        self.verbose = verbose
        if is_fitted_ is not None:
            self.is_fitted_ = is_fitted_
        transform_kwargs.update(self.fixed_transform_kwargs)
        self.set_transform(kwargs=transform_kwargs)

    def __call__(self, X: Chem.Mol|list[Chem.Mol]) -> np.ndarray:
        """
        Apply the tokenizer to the input data.

        Parameters:
        ----------
        X : Chem.Mol or list[Chem.Mol]
            The input data to be tokenized. Can be a single RDKit molecule or a list of molecules.
            
        Returns:
        -------
        np.ndarray
            The tokenized representation of the input data.
        """
        if not self.is_fitted_:
            raise ValueError('Tokenizer must be fit before calling.')
        X = self.transform(X)
        return X
    
    
    def save_transform(self, X: Chem.Mol|list[Chem.Mol], path: str):
        """
        Transform and save the data to a file.

        Parameters:
        ----------
        X : Chem.Mol or list[Chem.Mol]
            The input data to be transformed and saved. Can be a single RDKit molecule or a list of molecules.
        path : str
            The path where the transformed data will be saved.
        """
        X = self.transform(X)
        torch.save(X, path)

    def set_transform(self, kwargs):
        """
        Set the transformation function for the tokenizer.

        Parameters:
        ----------
        kwargs : dict
            Keyword arguments for the transformation function.
        """
        self.transform_kwargs = kwargs
        self.transform = self._transform_base(**kwargs)
        

    def _transform_base(self, **kwargs):
        """
        The base transformation function.
        This method should be overridden by subclasses to implement the actual transformation logic.
        """
        raise NotImplementedError

    def fit(self, mols: Chem.Mol, y: Optional[np.ndarray] = None) -> None:
        """
        Fit the tokenizer to the input data.

        Parameters:
        ----------
        mols : Chem.Mol or list[Chem.Mol]
            The input data to fit the tokenizer on. Can be a single RDKit molecule or a
            list of molecules.
        y : Optional[np.ndarray], optional
            Optional target values. Not used in this base class, but can be used in subclasses
            for supervised learning tasks. Defaults to None.
        
        Returns:
        -------
        BaseTokenizer
            Returns the fitted tokenizer instance.
        """
        self.is_fitted_ = True
        return self

    @property
    def name(self) -> str:
        """
        Get the name of the tokenizer.
        """
        return self.__class__.__name__
    
    def to_dict(self) -> dict:
        """
        Convert the tokenizer to a dictionary representation.
        Useful for saving the tokenizer's parameters and state.
        Overrides the default `to_dict` method to include additional information.

        Returns:
        -------
        dict
            A dictionary containing the tokenizer's name, fitted status, and transformation parameters.
        """
        return {
            'name': self.name,
            'is_fitted_': self.is_fitted_,
            'transform_kwargs': self.transform_kwargs,
        }
    
    def save(self, path: str, params_only: bool = False):
        """
        Save the tokenizer to a file. Uses PyTorch's `torch.save` method.

        Parameters:
        ----------
        path : str
            The path where the tokenizer will be saved.
        params_only : bool, optional
            If True, the tokenizer will be saved as a dictionary.
            If False, the entire tokenizer object will be saved. Defaults to False.
        """
        if params_only:
            params = self.to_dict()
            torch.save(params, path)
        else:
            torch.save(self, path)

    def raw(self, X: Chem.Mol):
        """
        Raw transformation of the data.

        Optionally implement raw and encode methods to allow for splitting of tokenization into
        two steps.
        Only some tokenizers will implement this.

        Useful for tokenizers that alter the encoding of the data depending on the vocabulary
        of the training data.

        E.g. AtomGraphTokenizer
            Raw method is used to generate graphs with atomic numbers as node features.
            Encode method is used to convert the raw graph into a tokenized graph by mapping atomic
            numbers to atom indices in a vocabulary.
            Ensures that new splits don't include atoms not in the training data.
        """
        raise NotImplementedError
    
    def encode(self, X):
        """
        Encode raw data.

        Optionally implement raw and encode methods to allow for splitting of tokenization into
        two steps.
        Only some tokenizers will implement this.

        Useful for tokenizers that alter the encoding of the data depending on the vocabulary
        of the training data.

        E.g. AtomGraphTokenizer
            Raw method is used to generate graphs with atomic numbers as node features.
            Encode method is used to convert the raw graph into a tokenized graph by mapping atomic
            numbers to atom indices in a vocabulary.
            Ensures that new splits don't include atoms not in the training data.
        """
        raise NotImplementedError
    
    def save_raw(self, X: Chem.Mol, path: str):
        """
        Save the raw data to a file.

        Parameters:
        ----------
        X : Chem.Mol or list[Chem.Mol]
            The input data to be saved. Can be a single RDKit molecule or a list of molecules.
        path : str
            The path where the raw data will be saved.
        """
        X = self.raw(X)
        torch.save(X, path)

    def __sklearn_is_fitted__(self):
        """
        Define the `__sklearn_is_fitted__` method to check if the tokenizer is fitted.
        """
        return self.is_fitted_
    
    def preprocess(self, mols: list[Chem.Mol]):
        """
        Optionally implement a preprocessing step for the data.
        This method can be used to preprocess the input data before tokenization.

        Parameters
        ----------
        mols : list[Chem.Mol]
            A list of RDKit molecule objects to be preprocessed.
        Returns
        -------
        list[Chem.Mol]
            A list of preprocessed RDKit molecule objects.
        """
        return mols

class BaseGraph:
    """
    Base class for graph transformations.

    This class provides a framework for converting molecules into graph representations.

    Parameters
    ----------
    node_types : dict, optional
        A dictionary mapping node types (e.g., atomic numbers) to integer indices.
        Defaults to a dictionary with a single entry for 'UNK' (unknown node type).
    edge_types : dict, optional
        A dictionary mapping bond types (e.g., single, double, aromatic) to integer indices
        Defaults to a dictionary with entries for common bond types.
    max_vocab_size : int, optional
        The maximum vocabulary size for node types. If specified, the tokenizer will limit
        the number of unique node types to this size.
    verbose : bool, optional
        If True, prints additional information during processing. Defaults to False.
    global_token : bool, optional
        If True, adds a global token to the graph. Defaults to False.
    """
    edge_types = {
        1.0: 0,
        2.0: 1,
        1.5: 2,
        3.0: 3,
    }
    node_types: dict = {'UNK': 0}

    def __init__(
        self, node_types: dict = None, edge_types: dict = None,
        max_vocab_size: int = None, verbose: bool = False, global_token: bool = False
    ):
        self.node_types = node_types or self.node_types
        self.edge_types = edge_types or self.edge_types
        self.max_vocab_size = max_vocab_size
        self.verbose = verbose
        self.global_token = global_token

    def get_edges(self, mol: Chem.Mol):
        """
        Get the edges for a molecule.

        Parameters:
        ----------
        mol : Chem.Mol
            RDKit molecule object.

        Returns:
        -------
        edge_index : torch.Tensor
            A tensor of shape (2, number of bonds*2) containing the edge indices.
            Each column represents an edge from a start atom to an end atom.
        edge_attr : torch.Tensor
            A tensor of shape (num_edges, 1) containing the edge attributes.
            Each row corresponds to an edge and contains the bond type as an integer index.
        """
        num_edges = mol.GetNumBonds()
        edge_index = torch.full(
            (2, num_edges*2), fill_value=-1, dtype=torch.long
        )
        edge_attr = torch.full(
            (num_edges*2, 1), fill_value=-1, dtype=torch.int
        )
        # loop over bonds and set edge index and edge attributes
        for bond in mol.GetBonds():
            # get start and end atom indices
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_type = self.edge_types[bond.GetBondTypeAsDouble()]

            # set edge index
            edge_index[0, bond.GetIdx()] = start
            edge_index[1, bond.GetIdx()] = end

            # set reverse edge index
            edge_index[0, bond.GetIdx() + num_edges] = end
            edge_index[1, bond.GetIdx() + num_edges] = start
            
            # set bond types
            edge_attr[bond.GetIdx()] = bond_type
            edge_attr[bond.GetIdx() + num_edges] = bond_type

        return edge_index, edge_attr
    
    def get_nodes(self, mol: Chem.Mol):
        """
        Get the raw node descriptor for an atom.
        """
        raise NotImplementedError
    
    def reset(self, mols: list[Chem.Mol|pyg.data.Data]):
        """
        Reset the node types.

        This method initializes the node types based on the input molecules.

        Parameters
        ----------
        mols : list[Chem.Mol|pyg.data.Data]
            A list of RDKit molecule objects or PyTorch Geometric Data objects.
            The method will determine the node types based on the unique atomic numbers
            or node features in the input molecules.

        Raises
        ------
        ValueError
            If the input molecules are not of the same type (either all RDKit molecules or all
            PyTorch Geometric Data objects).
        """
        node_types = {}
        if all(isinstance(x, Chem.Mol) for x in mols):
            mols = [self.raw(mol) for mol in mols if mol is not None]
        elif all(isinstance(x, pyg.data.Data) for x in mols):
            pass
        else:
            raise ValueError(
                'All molecules must be of the same type, either RDKit molecules '\
                'or PyTorch Geometric Data objects.'
            )
        batch = pyg.data.Batch.from_data_list(mols)
        assert torch.all(batch.raw), 'Data must be raw graphs.'
        x = batch.x
        unique_nodes, counts = torch.unique(x, return_counts=True)
        unique_nodes = unique_nodes[torch.argsort(counts, descending=True)]
    
        for node in unique_nodes:
            node_types[node.item()] = len(node_types)

        node_types['UNK'] = len(node_types)
        if self.global_token:
            num_tokens_per_node = x.size(1)
            for i in range(num_tokens_per_node):
                node_types[f'GLOBAL_{i}'] = len(node_types)
        self.node_types = node_types

    def add_global_token(self, graph: pyg.data.Data):
        """
        Add a global token to the graph.

        Parameters:
        ----------
        graph : pyg.data.Data
            The graph data to which the global token will be added.

        Returns:
        -------
        pyg.data.Data
            The graph data with the global token added.
        """
        if self.global_token:
            if 'x' not in graph:
                raise ValueError('Graph does not contain node features.')
            if graph.raw:
                raise ValueError('Graph must be encoded.')
            graph = graph.clone()
            num_tokens_per_node = graph.x.size(1)
            global_token = torch.empty(1, num_tokens_per_node, dtype=torch.long)
            for i in range(num_tokens_per_node):
                global_token[0, i] = self.node_types[f'GLOBAL_{i}']

            global_edges = torch.full((2, graph.num_nodes), fill_value=-1, dtype=torch.long)
            for i in range(graph.num_nodes):
                global_edges[0, i] = i
                global_edges[1, i] = graph.num_nodes

            graph.x = torch.cat([graph.x, global_token], dim=0)
            graph.edge_index = torch.cat([graph.edge_index, global_edges], dim=1)
            graph.global_idx = graph.x.size(0) - 1
            
        return graph
    
    @property
    def empty_graph(self):
        raise NotImplementedError(
            "empty_graph must be implemented in subclasses. \
            Method for handling None inputs.")
        
    def raw(self, mol: Chem.Mol):
        """
        Generate the raw graph data for a molecule.
        
        Parameters
        ----------
        mol : Chem.Mol
            The molecule to generate the graph for.
            
        Returns
        -------
        pyg.data.Data
            The raw graph data.
        """
        if mol is None:
            return self.empty_graph
        
        x = self.get_nodes(mol)
        # initialize edge index and edge attributes
        edge_index, edge_attr = self.get_edges(mol)

        return pyg.data.Data(
            x=x, edge_index=edge_index, edge_attr=edge_attr,
            raw=True, empty=torch.tensor([False]),
        )
    
    def encode(self, graph: pyg.data.Data):
        """
        Tokenize a raw graph.
        
        Parameters
        ----------
        graph : pyg.data.Data
            The raw graph data. (output of `BaseGraph.raw`)
        
        Returns
        -------
        pyg.data.Data
            The encode graph data.
        """
        if not graph.raw:
            raise ValueError('Graph must be raw.')
        if 'x' not in graph:
            return graph
        graph = graph.clone()
        unk = self.node_types['UNK']
        for i in range(graph.x.size(0)):
            for j in range(graph.x.size(1)):
                graph.x[i, j] = self.node_types.get(int(graph.x[i, j]), unk)
        graph.raw = False
        if len(graph.x) > 0:
            graph = self.add_global_token(graph)

        return graph

    def transform(self, mol: Chem.Mol|pyg.data.Data):
        """
        Make a graph from a molecule.

        Parameters
        ----------
        mol : Chem.Mol
            RDKit molecule object.
        
        Returns
        -------
        pyg.data.Data
            The tokenized graph data.
        """
        if isinstance(mol, pyg.data.Data):
            return self.encode(mol)
        else:
            graph = self.raw(mol)
            return self.encode(graph)
    
    def __call__(
            self,
            X: Chem.Mol|pyg.data.Data|list[Chem.Mol|pyg.data.Data]
        ) -> pyg.data.Data|list[pyg.data.Data]:
        """
        Apply the transformation to a single molecule or a list of molecules.

        Parameters
        ----------
        X : Chem.Mol or pyg.data.Data or list[Chem.Mol|pyg
            data.Data]
            The input data to be transformed. Can be a single RDKit molecule, 
            a PyTorch Geometric Data object, or a list of molecules or graphs.
        
        Returns
        -------
        pyg.data.Data or list[pyg.data.Data]
            The transformed graph data. If a single molecule is provided, returns a single 
            `pyg.data.Data` object.
            If a list of molecules is provided, returns a list of `py

        """
        if isinstance(X, Chem.Mol|pyg.data.Data|None):
            return self.transform(X)
        
        assert all(isinstance(m, Chem.Mol|pyg.data.Data|None) for m in X)

        out = []
        pbar = tqdm(total=len(X), desc='Generating graphs', disable=not self.verbose)
        for idx, mol in enumerate(X):
            out.append(self.transform(mol))
            pbar.update(1)
        pbar.close()
        return out


class GraphTokenizer(BaseTokenizer):
    """
    Class to tokenize molecules into graphs using atom types and bond types.

    Parameters
    ----------
    transform_kwargs : dict
        Keyword arguments for the BaseGraph.
    verbose : bool
        Whether to print progress information.
    """
    def __init__(
        self,
        transform_kwargs: dict = {},
        verbose: bool = False,
        **kwargs,
    ):
        super(GraphTokenizer, self).__init__(
            transform_kwargs=transform_kwargs, verbose=verbose
        )
    
    def raw(self, mol: Chem.Mol):
        """
        Generate the raw graph data for a molecule.

        Parameters:
        ----------
        mol : Chem.Mol
            The molecule to generate the graph for.

        Returns:
        -------
        pyg.data.Data
            The raw graph data.
        """
        return self.transform.raw(mol)
    
    def encode(self, graph: pyg.data.Data):
        """
        Encode a raw graph.

        Parameters:
        ----------
        graph : pyg.data.Data
            The raw graph data. (output of `BaseGraph.raw`)
        
        Returns:
        -------
        pyg.data.Data
            The encoded graph data.
        """
        return self.transform.encode(graph)
    
    def fit(self, mols: list[Chem.Mol|pyg.data.Data], y: None = None) -> None:
        """
        Fit the tokenizer to the provided molecules.

        Parameters:
        ----------
        mols : list[Chem.Mol|pyg.data.Data]
            A list of RDKit molecule objects or PyTorch Geometric Data objects to be tokenized.
        y : None, optional
            An optional array of labels corresponding to the molecules. Not used in this base class,
            but can be used in subclasses for supervised learning tasks. Defaults to None.

        Returns:
        -------
        GraphTokenizer
            Returns the fitted tokenizer instance.
        """
        self = super().fit(mols, y)
        self.transform.reset(mols)
        return self

    @property
    def vocab_size(self):
        """
        How many node types are there in the vocabulary.

        Returns:
        -------
        int
            The number of unique node types in the vocabulary.
        """
        return len(self.transform.node_types)
    
    @property
    def edge_types(self):
        """
        The edge types used in the graph transformation.

        Returns:
        -------
        dict
            A dictionary mapping edge types to integer indices.
        """
        return self.transform.edge_types

    @property
    def node_types(self):
        """
        The node types used in the graph transformation.

        Returns:
        -------
        dict
            A dictionary mapping node types (e.g., atomic numbers) to integer indices.
        """
        return self.transform.node_types
    
    def to_dict(self):
        """
        Convert the tokenizer's parameters and state to a dictionary.
        """
        params = super().to_dict()
        params['transform_kwargs']['node_types'] = self.node_types
        params['transform_kwargs']['edge_types'] = self.edge_types
        params['transform_kwargs']['max_vocab_size'] = self.transform.max_vocab_size
        return params

    def preprocess(self, mols):
        """
        Preprocess the input molecules by generating raw graphs.

        Parameters:
        ----------
        mols : list[Chem.Mol]
            A list of RDKit molecule objects to be preprocessed.
        
        Returns:
        -------
        list[pyg.data.Data]
            A list of raw graph data objects, where each object corresponds to a molecule.
            The graphs are generated using the `raw` method of the BaseGraph.
        """
        return [self.transform.raw(m) for m in mols]