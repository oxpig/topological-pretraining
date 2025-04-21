import numpy as np
from _src.nn import GIN, RegressionHead, BinaryHead
from sklearn.base import BaseEstimator
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch_geometric as pyg

from tqdm import tqdm
from typing import Callable, Literal

class GraphDatasetFromList(torch.utils.data.Dataset):
    def __init__(self, data: list[pyg.data.Data], y: np.ndarray):
        for i, d in enumerate(data):
            d.y = y[i]
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class SklearnGIN(torch.nn.Module, BaseEstimator):

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
        gnn_kwargs: dict = {'train_eps': True},
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
        lr_scale=1.0, lr_half_life=None,
        lr=None,
        weight_decay=0.0,
        head_layers=1, head_hidden_dim=None,
        return_loss=False,
        neptune_run=None,
        neptune_location='model_loss',
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
        self.neptune_location = neptune_location
        self.neptune_run = neptune_run
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

        if self.neptune_run:
            self.neptune_run[f'{neptune_location}/num_params'].append(sum([p.numel() for p in self.parameters()]))
            self.neptune_run[f'{neptune_location}/lr'].append(self.lr)
            self.neptune_run[f'{neptune_location}/vocab_size'].append(vocab_size)

    def forward(self, x):
        x = self.gnn(**x)['global_state']
        return self.head(x)
    
    def embed(self, X: list[pyg.data.Data]):
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
                    if self.neptune_run is not None:
                        self.neptune_run[f'{self.neptune_location}/batch_loss'].append(loss.item())
                    loss.backward()
                    optimizer.step()
                    losses[i, j] = loss.item()
                    if lr_scheduler:
                        lr_scheduler.step()
                    pbar.set_description(f'Epoch {i+1}/{self.epochs} | Batch loss: {loss.item()}')
                    pbar.update()
                if i != self.epochs - 1:
                    pbar.reset()
                
                if self.neptune_run:
                    epoch_mean_loss = losses[i].mean()
                    self.neptune_run[f'{self.neptune_location}/epoch_loss'].append(epoch_mean_loss)
            
            if self.return_loss:
                return losses

    def predict(self, X: list[pyg.data.Data]):
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
        weights = compute_class_weight(y=y, classes=np.unique(y), class_weight='balanced')
        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
        weights = weights.unsqueeze(-1).unsqueeze(-1)
        return weights



