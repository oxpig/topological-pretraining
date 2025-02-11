from .mlp import MLP
from .gnn import BaseGNN

import torch
import torch_geometric as pyg

class GINLayer(pyg.nn.conv.GINConv):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_norm: bool = False,
        act: str = 'relu',
        final_act: str = None,
        eps: float = 0.0,
        train_eps: bool = False,
        **kwargs,
    ):
        mlp = MLP(
            input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim,
            num_layers=num_layers, dropout=dropout, batch_norm=batch_norm,
            act=act, final_act=final_act
        )
        super(GINLayer, self).__init__(
            nn=mlp, eps=eps, train_eps=train_eps
        )

class GIN(BaseGNN):

    def _init_layer(self, input_dim, output_dim, **kwargs):

        if 'act' not in kwargs:
            kwargs['act'] = self.act_type
        if 'dropout' not in kwargs:
            kwargs['dropout'] = self.dropout_value
        if 'batch_norm' not in kwargs:
            kwargs['batch_norm'] = self.use_batch_norm

        return GINLayer(input_dim, output_dim, **kwargs)
        

            