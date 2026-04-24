import numpy as np
from _src.nn import GIN, RegressionHead, BinaryHead
from sklearn.base import BaseEstimator
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch_geometric as pyg

from tqdm import tqdm
from typing import Callable, Literal

class GraphDatasetFromList(torch.utils.data.Dataset):
    """
    A dataset class for creating a PyTorch Geometric dataset from a list of graphs.

    Parameters:
    ----------
    data : list[pyg.data.Data]
        A list of PyTorch Geometric Data objects representing the graphs.
    y : np.ndarray
        Target values for the graphs, should match the length of `data`.
    """
    def __init__(self, data: list[pyg.data.Data], y: np.ndarray):
        for i, d in enumerate(data):
            d.y = y[i]
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class SklearnGIN(torch.nn.Module, BaseEstimator):
    """
    A sklearn wrapper for the GIN model for molecular graphs using PyTorch Geometric.
    This class allows the GIN model to be used in a sklearn-like fashion, including
    methods for fitting, predicting, and transforming data.

    GIN model for molecular graphs using PyTorch Geometric, compatible with sklearn.

    Parameters:
    ----------
    input_dim : int
        The input dimension.
        If using tokenization, this is the number of tokens per node.
        Otherwise, this is the number of features per node.
    hidden_dim : int
        The hidden dimension of the GIN model.
    task : Literal['classification', 'regression']
        The type of task to perform, either 'classification' or 'regression'.
    vocab_size : int, optional
        The size of the vocabulary for node embeddings, if applicable.
    node_embedding_dim : int, optional
        The dimension of the node embeddings, if applicable.
    gnn_layers : int, optional
        The number of GNN layers in the GIN model. Default is 1.
    dropout : float, optional
        The dropout rate for the GIN model. Default is 0.0.
    batch_norm : bool, optional
        Whether to use batch normalization in the GIN model. Default is False.
    act : str, optional
        The activation function to use in the GIN model. Default is 'relu'.
    layer_pool_type : slice|int|Literal['last', 'sum', 'mean', max', 'concat'], optional
        The type of pooling to apply at the layer level. Default is 'concat'.
        `last` uses the last layer output, `sum` sums all layer outputs, 
        `mean` averages all layer outputs, `max` takes the maximum of all layer outputs,
        and `concat` concatenates all layer outputs.
    graph_pool_type : Literal[None, 'sum', 'mean', 'max', 'global_node'], optional
        The type of pooling to apply at the graph level. Default is 'max'.
        If `global_node` is specified, a global node is added to the graph.
    train_eps : bool, optional
        Whether to train the epsilon parameters in the GIN model. Default is True.
    eps : float, optional
        The initial value for the epsilon parameters in the GIN model. Default is 0.
    mlp_layers : int, optional
        The number of layers in the MLP head. Default is 1.
    weight_init : Callable|Literal['standard', 'xavier_uniform', 'xavier_normal',
        'normal', 'zeros', 'ones'], optional
        The weight initialization method for the GIN model. Default is 'standard'.
    bias_init : Callable|Literal['standard', 'xavier_uniform', 'xavier_normal',
        'normal', 'zeros', 'ones'], optional
        The bias initialization method for the GIN model. Default is 'standard'.
    share_weights : bool, optional
        Whether to share weights across GIN layers. Default is False.
        True means all GIN layers share the same weights.
        This is useful for training a model with multiple convolutions on a small dataset.
    epochs : int, optional
        The number of epochs to train the model. Default is 50.
    batch_size : int, optional
        The batch size for training. Default is 32.
    lr : float, optional
        The learning rate for the optimizer. If None, it is calculated based on `lr_scale`.
    lr_scale : float, optional
        The learning rate scale factor. Default is 1.0.
        The learning rate is calculated as `lr_scale * (num_params ** -0.5)`,
        where `num_params` is the number of parameters in the model.
    lr_half_life : float, optional
        The half-life for the learning rate decay. Default is None.
        If specified, the learning rate will decay by half every `lr_half_life` epochs.
    weight_decay : float, optional
        The weight decay for the optimizer. Default is 0.0.
    head_layers : int, optional
        The number of layers in the prediction head of the model. Default is 1.
    head_hidden_dim : int, optional
        The hidden dimension of the prediction head. If None, it defaults to `hidden_dim`.
    return_loss : bool, optional
        If True, the `fit` method returns the training losses. Default is False.
    logging : dict, optional
        A dictionary to store logging information such as losses and hyperparameters. Default is None.
    verbose : bool, optional
        If True, enables verbose output during training. Default is False.
    device : str, optional
        This is for compatibility.
    **kwargs : Any
        Additional keyword arguments for the GIN model.
    """
    def __init__(
        self, input_dim: int, hidden_dim: int,
        task: Literal['classification', 'regression'],
        vocab_size: int = None,
        node_embedding_dim: int = None,
        gnn_layers: int = 1,
        dropout: float = 0.0, batch_norm: bool = False,
        act: str = 'relu',
        layer_pool_type: slice|int|Literal[
            'last', 'sum', 'mean',
            'max', 'concat'
        ] = 'concat',
        graph_pool_type: Literal[
            None, 'sum', 'mean',
            'max', 'global_node'
        ] = 'max',
        train_eps: bool = True,
        eps: float = 0.0,
        mlp_layers: int = 1,
        weight_init: Callable|Literal[
            'standard', 'xavier_uniform', 'xavier_normal',
            'normal', 'zeros', 'ones',
        ] = 'standard',
        bias_init: Callable|Literal[
            'standard', 'xavier_uniform', 'xavier_normal',
            'normal', 'zeros', 'ones',
        ] = 'standard',
        share_weights: bool = False,
        epochs=50, batch_size=32,
        lr=None,
        lr_scale=1.0, lr_half_life=None,
        weight_decay=0.0,
        head_layers=1, head_hidden_dim=None,
        return_loss=False,
        logging=None,
        verbose=False,
        device='cpu', # for compatibility with other models, uses cuda if available
        **kwargs
    ):
        super(SklearnGIN, self).__init__()
        gnn_kwargs = {
            'train_eps': train_eps,
            'eps': eps,
            'num_layers': mlp_layers,
            'weight_init': weight_init,
            'bias_init': bias_init,
        }
        self.gnn = GIN(
            input_dim=input_dim, hidden_dim=hidden_dim,
            node_embedding=(vocab_size, node_embedding_dim),
            num_layers=gnn_layers,
            dropout=dropout, batch_norm=batch_norm, act=act,
            layer_pool_type=layer_pool_type,
            graph_pool_type=graph_pool_type,
            share_weights=share_weights,
            gnn_kwargs=gnn_kwargs,
            **kwargs
        )
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.node_embedding_dim=node_embedding_dim
        self.vocab_size=vocab_size
        self.gnn_layers=gnn_layers
        self.dropout=dropout
        self.batch_norm=batch_norm
        self.act=act
        self.layer_pool_type=layer_pool_type
        self.graph_pool_type=graph_pool_type
        self.share_weights=share_weights
        self.gnn_kwargs=gnn_kwargs

        self.task = task
        self.head_layers = head_layers
        self.head_hidden_dim = head_hidden_dim

        self.epochs = epochs
        self.batch_size = batch_size
        self.lr_scale = lr_scale
        self.lr_half_life = lr_half_life
        self.weight_decay = weight_decay
        self.logging = logging if logging is not None else {}
        self.return_loss = return_loss
        self.verbose = verbose
        self.class_weights = None

        if task == 'classification':
            self.loss_fn = torch.nn.BCELoss()
            self.head = BinaryHead(
                input_dim=self.gnn.out_shape,
                hidden_dim=head_hidden_dim if head_hidden_dim != None else hidden_dim,
                output_dim=1,
                num_layers=head_layers,
                act=act,
                dropout=dropout, batch_norm=batch_norm,
            )
            
        elif task == 'regression':
            self.loss_fn = torch.nn.MSELoss()
            self.head = RegressionHead(
                input_dim=self.gnn.out_shape,
                hidden_dim=head_hidden_dim if head_hidden_dim != None else hidden_dim,
                output_dim=1,
                num_layers=head_layers,
                act=act,
                dropout=dropout, batch_norm=batch_norm,
            )
        else:
            raise ValueError('Invalid target type')
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        if lr is None:
            self.lr = self.lr_scale*sum([p.numel() for p in self.parameters()]) **-0.5
        else:
            self.lr = lr

        if self.logging is not None:
            num_params = sum([p.numel() for p in self.parameters()])
            self.logging['num_params'] = num_params
            self.logging['lr'] = self.lr
            self.logging['vocab_size'] = vocab_size

    def forward(self, x):
        """
        Forward pass through the GIN model and the prediction head.

        Parameters:
        ----------
        x : pyg.data.Data | pyg.data.Batch
            Input data for the GIN model, should be a PyTorch Geometric Data object
            or a batch of Data objects.
        
        Returns:
        -------
        torch.Tensor
            The output of the prediction head after processing the GIN model output.
        """
        x = self.gnn(**x)['global_state']
        return self.head(x)
    
    def embed(self, X: list[pyg.data.Data]):
        """
        Get the graph embeddings from the GIN model for a list of graphs.

        Parameters:
        ----------
        X : list[pyg.data.Data]
            A list of PyTorch Geometric Data objects representing the graphs.

        Returns:
        -------
        np.ndarray
            The graph embeddings, shape is (num_graphs, out_shape).
            `out_shape` is the output shape of the GIN model.
        
        This method processes each graph in the list, passing it through the GIN model
        to obtain the global state (embedding) of each graph. The embeddings are then
        collected and returned as a NumPy array.
        """
        self.eval()
        data = GraphDatasetFromList(X, np.zeros(len(X)))
        preds = np.zeros((len(X), self.gnn.out_shape))
        for i, graph in enumerate(data):
            graph = graph.to(self.device)
            graph = graph.to(self.device)
            out = self.gnn(**graph)['global_state']
            preds[i] = out.detach().cpu().numpy()
        return preds

    def fit(self, X: list[pyg.data.Data], y):
        """
        Fit the GIN model to the input data and target values.

        Parameters:
        ----------
        X : list[pyg.data.Data]
            A list of PyTorch Geometric Data objects representing the graphs.
        y : np.ndarray
            Target values for the graphs, should match the length of `X`.

        Returns:
        -------
        np.ndarray | None
            If `return_loss` is True, returns the training losses as a 2D NumPy array
            with shape (epochs, num_batches). Otherwise, returns None.
        """
        if self.task == 'classification':
            self.class_weights = self.cal_class_weights(y)
            self.head.set_class_weight(self.class_weights)
    
        self.train()
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        data = GraphDatasetFromList(X, y)
        # use drop_last=True if the last batch contains only one sample
        if len(data) % self.batch_size == 1: drop_last = True
        else: drop_last = False
        loader = pyg.loader.DataLoader(
            data, batch_size=self.batch_size, shuffle=True,
            drop_last=drop_last,
        )
        losses = np.zeros((self.epochs, len(loader)))
        lr_scheduler = None
        if self.lr_half_life is not None:
            gamma = 0.5 ** (1 / (self.lr_half_life * len(loader)))
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=gamma,
            )
        if self.logging is not None:
            self.logging['batch_loss'] = []
            self.logging['epoch_loss'] = []

        with tqdm(
            total=len(loader),
            disable=not self.verbose,
            desc=f'Epoch 1/{self.epochs} | Batch loss: {np.nan}',
        ) as pbar:
            for i in range(self.epochs):
                for j, batch in enumerate(loader):
                    optimizer.zero_grad()
                    batch = batch.to(self.device)
                    out = self(batch)
                    y = batch.y.unsqueeze(1)
                    loss = self.head.loss(y=y, pred=out)
                    if self.logging is not None:
                        self.logging['batch_loss'].append(loss.item())
                    loss.backward()
                    optimizer.step()
                    losses[i, j] = loss.item()
                    if lr_scheduler:
                        lr_scheduler.step()
                    pbar.set_description(f'Epoch {i+1}/{self.epochs} | Batch loss: {loss.item()}')
                    pbar.update()
                if i != self.epochs - 1:
                    pbar.reset()
                
                if self.logging is not None:
                    epoch_mean_loss = losses[i].mean()
                    self.logging['epoch_loss'].append(epoch_mean_loss)
            
            if self.return_loss:
                return losses

    def predict(self, X: list[pyg.data.Data]):
        """
        Predict using the fitted GIN model on a list of graphs.

        Parameters:
        ----------
        X : list[pyg.data.Data]
            A list of PyTorch Geometric Data objects representing the graphs.

        Returns:
        -------
        np.ndarray
            The predicted values for the graphs, shape is (num_graphs, 1).
            If the task is 'classification', the output is a probability score.
            If the task is 'regression', the output is a continuous value.
        """
        self.eval()
        data = GraphDatasetFromList(X, np.zeros(len(X)))
        loader = pyg.loader.DataLoader(
            data, batch_size=self.batch_size, shuffle=False,
        )
        preds = []
        for batch in loader:
            batch = batch.to(self.device)
            out = self(batch)
            preds.append(out.detach().cpu().numpy())
        return np.concatenate(preds).flatten()

    def cal_class_weights(self, y: np.ndarray):
        """
        Calculate class weights for the classification task.
        This method uses sklearn's compute_class_weight to calculate the weights
        for each class based on the target values.

        Parameters:
        ----------
        y : np.ndarray
            Target values for the graphs, should be a 1D array of class labels.

        Returns:
        -------
        torch.Tensor
            A tensor containing the class weights, shape is (num_classes, 1).
            The class weights are calculated to balance the classes in the dataset.
        """
        weights = compute_class_weight(y=y, classes=np.unique(y), class_weight='balanced')
        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
        weights = weights.unsqueeze(-1).unsqueeze(-1)
        return weights



