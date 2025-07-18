from _src.featurization.base import BaseFeaturizer
from _src.featurization.load import read_from_dict
from _src.nn.pred_head import PredHead
from _src.nn import get_nn

import numpy as np
from rdkit import Chem
import torch
import torch_geometric as pyg
from typing import Literal

class PreTrainedModel(torch.nn.Module):
    """
    Wrapper for a pre-trained model that transforms molecules into embeddings.
    Model can be loaded from a file or dictionary.

    Parameters:
    ----------
    path : str | None
        Path to the pre-trained model file.
    params : dict | None
        Dictionary containing the model details.
        Includes
        - featurizer: dict 
            Featurizer details. See _src.featurizers.base.BaseFeaturizer.to_dict for details.
        - main: dict
            Main model details.
            - cls: class name of the main model.
            - kwargs: keyword arguments for the main model.
            - state: state dictionary of the main model.
        - heads: dict[str, dict]
            Prediction heads details.
            For each head:
                - cls: class name of the head model
                - kwargs: keyword arguments for the head model.
                - state: state dictionary of the head model.
    device : str
        Device to load the model onto. Default is 'cuda' if available, otherwise 'cpu'.
    asarray : bool
        Whether to return the output as a NumPy array. Default is True.
        If False, the output will be a PyTorch tensor.
    """
    def __init__(
        self,
        path: str|None = None,
        params: dict|None = None,
        device: str = None,
        asarray: bool = True,
    ):
        super(PreTrainedModel, self).__init__()
        self.asarray = asarray
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        self.device = device
        if params is not None:
            self.from_dict(params)
        elif path is not None:
            self.load(path=path)
        else:
            raise ValueError('Either path or params must be provided.')
        self.to_device(self.device)
        
    def from_dict(self, params: dict):
        """
        Load the model from a dictionary of parameters.

        Parameters:
        ----------
        params : dict
            Dictionary containing the model details.
            Includes:
            - featurizer: dict 
                Featurizer details. See _src.featurizers.base.BaseFeaturizer.to_dict for details.
            - main: dict
                Main model details.
                - cls: class name of the main model.
                - kwargs: keyword arguments for the main model.
                - state: state dictionary of the main model.
            - heads: dict[str, dict]
                Prediction heads details.
                For each head:
                    - cls: class name of the head model
                    - kwargs: keyword arguments for the head model.
                    - state: state dictionary of the head model.
        """
        featurizer = params.pop('featurizer')
        self.featurizer = read_from_dict(featurizer)
        main_model = params.pop('main')
        main_cls = main_model['cls']
        main_cls = get_nn(main_cls)
        self.model = main_cls(**main_model['kwargs'])
        self.model.load_state_dict(main_model['state'])
        self.model.eval()
        self.heads = torch.nn.ModuleDict()
        self.heads_kwargs = {}
        for head in params.get('heads', {}):
            head_cls = params['heads'][head]['cls']
            head_cls = get_nn(head_cls)
            head_kwargs = params['heads'][head]['kwargs']
            head_state = params['heads'][head]['state']
            self.heads[head] = head_cls(**head_kwargs)
            self.heads[head].load_state_dict(head_state)
            self.heads[head].eval()
            self.heads_kwargs[head] = head_kwargs

        self.to_device()

    def embed(self, x: Chem.Mol):
        """
        Embed the input RDKit molecule object into a tensor representation.
        Embedding is done without prediction heads.

        Parameters:
        ----------
        x : Chem.Mol
            RDKit molecule object to be embedded.

        Returns:
        -------
        torch.Tensor
            A tensor representation of the molecule.
            The shape of the tensor depends on the model architecture.
        """
        x = self.tokenize(x)
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.to(self.device)
        x = self.model(x)
        return x

    def forward(self, x: Chem.Mol):
        """
        Forward pass through the model to get the embeddings.
        
        Parameters:
        ----------
        x : Chem.Mol
            RDKit molecule object to be processed.

        Returns:
        -------
        torch.Tensor
            A tensor representation of the molecule after passing through the model.
            If `self.asarray` is True, the tensor will be converted to a NumPy array
            before returning.
        """
        with torch.no_grad():
            x = self.embed(x)
        if self.asarray:
            x = x.detach().cpu().numpy()
        return x
    
    def get_head_preds(
        self, x: Chem.Mol|list[Chem.Mol]
    ):
        """
        Get predictions from the model's prediction heads.

        Parameters:
        ----------
        x : Chem.Mol | list[Chem.Mol]
            RDKit molecule object or a list of RDKit molecule objects.

        Returns:
        -------
        dict
            A dictionary containing predictions for each head.
            The keys are the names of the heads, and the values are the predictions.
        """
        x = self.tokenize(x)
        x = self.model(x)
        preds = {}
        with torch.no_grad():
            for target in self.heads:
                preds[target] = self.heads[target](x)
            return preds
        
    @property
    def model_cls(self):
        """
        Get the class name of the main model.

        Returns:
        -------
        str
            The class name of the main model.
        """
        return self.model.__class__.__name__
    
    @property
    def model_state_dict(self):
        """
        Get the state dict of the main model.
        """
        return self.model.state_dict()
    
    def to_dict(self):
        """
        Convert the model and featurizer to a dictionary format.

        Returns:
        -------
        dict
            A dictionary containing the model and featurizer details.
        """
        params = {
            'featurizer': self.featurizer.to_dict(),
            'main': {
                'cls': self.model_cls,
                'kwargs': self.model_kwargs,
                'state': self.model_state_dict,
            },
            'heads': {}
        }
        for target in self.heads:
            params['heads'][target] = {
                'state': self.heads[target].state_dict(),
                'cls': self.heads[target].__class__.__name__,
                'kwargs': self.heads_kwargs[target],
            }
        return params

    def save(self, path):
        """
        Save the model and featurizer to a file.

        Parameters:
        ----------
        path : str
            Path to save the model and featurizer.
            The model will be saved in a dictionary format.
            The dictionary will include the featurizer, main model, and prediction heads.
            Saving uses `torch.save` to serialize the dictionary.
        """
        params = self.to_dict()
        torch.save(params, path)
        self.path = path

    def to_device(self, device = None):
        """
        Move the model and featurizer to the specified device.
        
        Parameters:
        ----------
        device : str | None
            Device to move the model and featurizer to.
            If None, the model will be moved to `self.device`.
        """
        if device is None:
            device = self.device
        else:
            self.device = device
        super().to(device)
        
    def load(self, path: str):
        """
        Load the model and featurizer from a file.

        Parameters:
        ----------
        path : str
            Path to the pre-trained model file.
            The file should contain a dictionary with the model and featurizer details.
        """
        self.path = path
        params = torch.load(path, weights_only=True, map_location='cpu')
        self.from_dict(params)

    def tokenize(self, X: Chem.Mol|list[Chem.Mol]):
        """
        Tokenize the input data into a format suitable for the model.
        
        Parameters:
        ----------
        X : Chem.Mol | list[Chem.Mol]
            RDKit molecule object or a list of RDKit molecule objects to be tokenized.
            
        Returns:
        -------
        Any
            Tokenized representation of the input data. E.g., pytorch geometric Data object or a tensor.
        """
        X = self.featurizer.transform(X)
        if isinstance(X, list):
            X = [x.to(self.device) for x in X]
        else:
            X = X.to(self.device)
        return X

class PreTrainedGNN(PreTrainedModel):
    """
    Wrapper for a pre-trained GNN model that transforms molecules into embeddings.

    Parameters:
    ----------
    path : str | None
        Path to the pre-trained GNN model file.
    params : dict | None
        Dictionary containing the GNN model details.
        See _src.featurizers.pretrained.PreTrainedModel.from_dict for details.
    embed_state : Literal['node', 'global', 'all']
        State to embed. Can be 'node', 'global', or 'all'.
        Option for changing the embedding state.
    layer_pool_type : slice | int | Literal['last', 'sum', 'mean', 'max', 'concat']
        Option for changing the layer pooling type.
        If None, the pooling type used in pre-training will be used.
    graph_pool_type : Literal['sum', 'mean', 'max', 'concat'] | None
        Option for changing the graph pooling type.
        If None, the pooling type used in pre-training will be used.
    device : str
        Device to load the model onto. Default is 'cuda' if available, otherwise 'cpu
    asarray : bool
        Whether to return the output as a NumPy array. Default is True.
        If False, the output will be a PyTorch tensor.
    """
    def __init__(
        self,
        path: str|None = None,
        params: dict|None = None,
        embed_state: Literal['node', 'global', 'all'] = 'global',
        layer_pool_type: slice|int|Literal['last', 'sum', 'mean', 'max', 'concat'] = None,
        graph_pool_type: Literal['sum', 'mean', 'max', 'concat']|None = None,
        device: str = None,
        asarray: bool = True,
        **kwargs,
    ):
        self.layer_pool_type = layer_pool_type
        self.graph_pool_type = graph_pool_type
        self.embed_state = embed_state
        super(PreTrainedGNN, self).__init__(path=path, params=params, device=device, asarray=asarray)
        self.to_device()
        
    def from_dict(self, params: dict):
        """
        Load the GNN model from a dictionary of parameters.

        Parameters:
        ----------
        params : dict
            Dictionary containing the GNN model details.
            Includes:
            - featurizer: dict
            - main: dict
                - cls: class name of the main model.
                - kwargs: keyword arguments for the main model.
                - state: state dictionary of the main model.
            - heads: dict[str, dict]
                Prediction heads details.
                For each head:
                    - cls: class name of the head model
                    - kwargs: keyword arguments for the head model.
                    - state: state dictionary of the head model.
        """
        super().from_dict(params)
        if self.layer_pool_type is not None:
            self.model.layer_pool_type = self.layer_pool_type
            self.model.out_shape = self.model.cal_out_shape()
        if self.graph_pool_type is not None:
            self.model.graph_pool_type = self.graph_pool_type
    
    def embed(
            self,
            X: Chem.Mol|pyg.data.Data|list[Chem.Mol|pyg.data.Data],
            embed_state: Literal['node', 'global', 'all'] = None
        ):
        """
        Embed the input data into a tensor representation using the GNN model.

        Parameters:
        ----------
        X : Chem.Mol | pyg.data.Data | list[Chem.Mol | pyg.data.Data]
            RDKit molecule object or a list of RDKit molecule objects 
            or PyTorch Geometric Data objects to be embedded.
        embed_state : Literal['node', 'global', 'all']
            State to embed. Can be 'node', 'global', or 'all'.
            If None, uses the `embed_state` set during initialization.

        Returns:
        -------
        torch.Tensor
            A tensor representation of the molecule.
            The shape of the tensor depends on the model architecture.
            If `self.asarray` is True, the tensor will be converted to a NumPy array
            before returning.
        """
        if embed_state is not None:
            self.embed_state = embed_state
        if isinstance(X, Chem.Mol|pyg.data.Data):
            X = [X]
        if all(isinstance(x, Chem.Mol|None) for x in X):
            X = self.tokenize(X)
        out_shape = (1, self.model.out_shape)
        out = []
        for x in X:
            if not torch.all(x.get("empty", False)):
                x = x.to(self.device)
                x = self.model(**x)
                if self.embed_state == 'node':
                    x = x['final_state']
                elif self.embed_state == 'global':
                    x = x['global_state']
                elif self.embed_state == 'all':
                    pass
                else:
                    raise ValueError(
                        f'Invalid embed_state {self.embed_state}. '\
                        f'Must be one of "node", "global", or "all".'
                    )
            else:
                x = torch.full(
                    size=out_shape, fill_value=torch.nan, device=self.device
                )
            out.append(x)
        if self.embed_state != 'all':
            out = torch.vstack(out)
        return out

    def initial_embed(
        self,
        X: Chem.Mol|pyg.data.Data|list[Chem.Mol|pyg.data.Data],
        keep_tokens=False,
    ):
        """
        Initial embedding of the input data before convolution.

        Parameters:
        ----------
        X : Chem.Mol | pyg.data.Data | list[Chem.Mol | pyg.data
        Data]
            RDKit molecule object or a list of RDKit molecule objects 
            or PyTorch Geometric Data objects to be embedded.
        keep_tokens : bool
            Whether to keep the tokens in the output. Default is False.
            If True, the output will include the tokens used for embedding.

        Returns:
        -------
        list[torch.Tensor]
            A list of tensors, each representing the initial embedding of a molecule.
            Each tensor will have shape (number of atoms, embedding size).
            If `keep_tokens` is True, the tokens used for embedding will be included in the output.
        """
        # embedding prior to convolution
        if isinstance(X, Chem.Mol|pyg.data.Data):
            X = [X]
        if all(isinstance(x, Chem.Mol|None) for x in X):
            X = self.tokenize(X)
        return [self.model.embed_graph_nodes(x, keep_tokens=keep_tokens) for x in X]
        
 
class PreTrainedFeaturizer(BaseFeaturizer):
    """
    Wrapper for a pre-trained model such that it can be used for featurization.

    Parameters:
    ----------
    transform_kwargs : dict
        gnn : bool
            Whether the model is a GNN. If True, uses PreTrainedGNN, otherwise uses PreTrainedModel.
        Additional keyword arguments for the model.
        See _src.featurizers.pretrained.PreTrainedModel and
        _src.featurizers.pretrained.PreTrainedGNN for details.
    """
    is_fitted_ = True
    precomputed = True

    def _transform_base(self, **kwargs):
        """
        Set the transformation function for the featurizer.
        
        Parameters:
        ----------
        **kwargs : dict, optional
            Additional keyword arguments for the transformation function.
            
        Returns:
        -------
        PreTrainedGNN | PreTrainedModel
            An instance of PreTrainedGNN or PreTrainedModel based on the `gnn` keyword argument.
            If `gnn` is True, returns PreTrainedGNN, otherwise returns PreTrainedModel.
        """
        if kwargs.get('gnn', False):
            return PreTrainedGNN(**kwargs)
        else:
            return PreTrainedModel(**kwargs)
    
    def to_dict(self):
        """
        Convert the featurizer to a dictionary format.
        
        Returns:
        -------
        dict
            A dictionary containing the featurizer details.
            Includes the parameters of the featurizer and the transformation function.
        """
        params = super().to_dict()
        params.update(self.transform.to_dict())
        return params
    
    def set_embed_state(
        self, embed_state: Literal['node', 'global', 'all']
    ):
        """
        Set the embedding state for the model.

        Parameters:
        ----------
        embed_state : Literal['node', 'global', 'all']
            State to embed. Can be 'node', 'global', or 'all'.
            This will change the state used for embedding in the model.
            If 'node', the model will return node embeddings.
            If 'global', the model will return global embeddings.
            If 'all', the model will return both node and global embeddings.
            Default is 'global'.
        """
        self.transform.embed_state = embed_state
        self.transform.asarray = False

    def preprocess(self, X: Chem.Mol|list[Chem.Mol]):
        """
        Transform the input data into the input format for the pre-trained model.

        Parameters:
        ----------
        X : Chem.Mol | list[Chem.Mol]
            RDKit molecule object or a list of RDKit molecule objects to be tokenized.

        Returns:
        -------
        list[Any]
            A list of pre-tokenized molecules. E.g., a list of PyTorch Geometric Data objects
            with tokenized nodes.
        """
        if isinstance(X, Chem.Mol):
            X = [X]
        X = self.transform.tokenize(X)
        return X
    
    def initial_embed(
        self, X: Chem.Mol|pyg.data.Data|list[Chem.Mol|pyg.data.Data],
        **kwargs
    ):
        """
        Initial embedding of the input data before convolution.
        
        Parameters:
        ----------
        X : Chem.Mol | pyg.data.Data | list[Chem.Mol | pyg.data.Data]
            RDKit molecule object or a list of RDKit molecule objects
            or PyTorch Geometric Data objects to be embedded.
        **kwargs : dict
            Additional keyword arguments for the initial embedding.
            These will be passed to the model's initial embedding function.
        
        Returns:
        -------
        list[Any]
            A list of tensors, each representing the initial embedding of a molecule.
            E.g., a list of PyTorch Geometric Data objects with feature vectors for each node.
        """
        # initial input embedding 
        # (i.e., tokens mapping to vectors of learned parameters)
        return self.transform.initial_embed(X, **kwargs)
    
    @property
    def device(self):
        """
        Get the device on which the model is loaded.
        
        Returns:
        -------
        str
            The device on which the model is loaded. E.g., 'cuda' or 'cpu'.
        """
        return self.transform.device