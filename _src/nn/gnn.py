from .mlp import act_fn

import torch
import torch_geometric as pyg
import torch_geometric
from typing import Literal

class BaseGNN(torch.nn.Module):

    use_edge_weight: bool = False
    use_edge_attr: bool = False

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        node_embedding: tuple[int,int] = None,
        padding_idx: int = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_norm: bool = False,
        act: str = 'relu',
        layer_pool_type: Literal['last', 'sum', 'mean', 'max', 'concat'] = 'concat',
        graph_pool_type: Literal[None, 'sum', 'mean', 'max',] = 'max',
        gnn_kwargs: dict = {},
        share_weights: bool = False,
        device: str = 'cpu',
    ):
        super(BaseGNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_value = dropout
        self.use_batch_norm = batch_norm
        self.act_type = act
        self.num_layers = num_layers
        self.layer_pool_type = layer_pool_type
        self.graph_pool_type = graph_pool_type
        self.gnn_kwargs = gnn_kwargs
        if 'act' not in gnn_kwargs:
            gnn_kwargs['act'] = act
        if 'dropout' not in gnn_kwargs:
            gnn_kwargs['dropout'] = dropout
        if 'batch_norm' not in gnn_kwargs:
            gnn_kwargs['batch_norm'] = batch_norm
        if node_embedding is not None:
            self.node_vocab_size = node_embedding[0]
            self.node_embedding_dim = node_embedding[1]
        else:
            self.node_vocab_size = None
            self.node_embedding_dim = None
        self.share_weights = share_weights
        self.layers = torch.nn.ModuleDict({
                'act': act_fn[act],
                'dropout': torch.nn.Dropout(dropout),
                'batch_norm': torch.nn.BatchNorm1d(hidden_dim) if batch_norm else torch.nn.Identity(),
            })
        self.num_hidden_states = self.num_layers
        if self.node_vocab_size is not None:
            self.layers['node_embedding'] = torch.nn.Embedding(
                self.node_vocab_size, self.node_embedding_dim, padding_idx=padding_idx
            )
            node_tokens = input_dim
            input_dim = self.node_embedding_dim*node_tokens
            if self.node_embedding_dim == self.hidden_dim:
                self.num_hidden_states += node_tokens
        else:
            self.layers['node_embedding'] = None

        layer_count = 0
        if input_dim != hidden_dim:
            self.layers[f'conv_{layer_count}'] = self._init_layer(
                input_dim=input_dim, hidden_dim=hidden_dim, **gnn_kwargs
            )
            layer_count += 1

        if share_weights:
            self.layers[f'conv_{layer_count}'] = self._init_layer(
                input_dim=hidden_dim, hidden_dim=hidden_dim, **gnn_kwargs
            )
            # note: only works for GIN-like models at the moment
            for i in range(layer_count+1, num_layers):
                self.layers[f'conv_{i}'] = self._init_layer(
                    mlp=self.layers[f'conv_{layer_count}'].nn,
                    **gnn_kwargs
                )

            self.shared_layer = f'conv_{layer_count}'
            layer_count += 1

        else:
            for i in range(layer_count, num_layers):
                self.layers[f'conv_{i}'] = self._init_layer(
                    hidden_dim, hidden_dim, **gnn_kwargs
                )

    def _init_layer(self, input_dim, output_dim, **kwargs):
        raise NotImplementedError
    
    def layer_pool(self, out: dict):
        hidden_states = out['hidden_states']
        node_embedding = out['node_embedding']
        if self.layer_pool_type == 'last':
            out = hidden_states[:, -1, :]
        elif self.layer_pool_type == 'sum':
            out = hidden_states.sum(dim=1)
            if node_embedding is not None:
                node_embedding = node_embedding.sum(dim=1)
                out = torch.cat((out, node_embedding), dim=-1)
        elif self.layer_pool_type == 'mean':
            out = hidden_states.mean(dim=1)
            if node_embedding is not None:
                node_embedding = node_embedding.mean(dim=1)
                out = torch.cat((out, node_embedding), dim=-1)

        elif self.layer_pool_type == 'max':
            out = hidden_states.max(dim=1).values
            if node_embedding is not None:
                node_embedding = node_embedding.max(dim=1).values
                out = torch.cat((out, node_embedding), dim=-1)

        elif self.layer_pool_type == 'concat':
            out = hidden_states.view(hidden_states.size(0), -1)
            if node_embedding is not None:
                node_embedding = node_embedding.view(node_embedding.size(0), -1)
                out = torch.cat((out, node_embedding), dim=-1)
        else:
            raise ValueError(
                f'Invalid layer pooling method: {self.layer_pool_type}\n'\
                'Valid options are: last, sum, mean, max, concat.'
            )
        return out

    
    def graph_pool(self, final_state: torch.Tensor, batch: torch.Tensor):
        if self.graph_pool_type is None:
            return None
        elif self.graph_pool_type == 'sum':
            return pyg.nn.pool.global_add_pool(final_state, batch)
        elif self.graph_pool_type == 'mean':
            return pyg.nn.pool.global_mean_pool(final_state, batch)
        elif self.graph_pool_type == 'max':
            return pyg.nn.pool.global_max_pool(final_state, batch)
        else:
            raise ValueError(
                f'Invalid graph pooling method: {self.graph_pool_type}\n'\
                'Valid options are: sum, mean, max.'
            )
    
    def forward(self, x, edge_index, edge_weight = None, edge_attr = None, batch = None):
        out = {
            'input': x,
        }
        state = 0
        num_hidden_states = self.num_hidden_states
        out['hidden_states'] = torch.zeros((x.size(0), num_hidden_states, self.hidden_dim)).to(x.device)
        if self.node_vocab_size is not None:
            x = self.layers['node_embedding'](x)
            if self.node_embedding_dim == self.hidden_dim:
                out['hidden_states'][:, :x.size(1), :] = x
                out['node_embedding'] = None
                state += x.size(1)
            else:
                out['node_embedding'] = x
            x = x.view(x.size(0), -1)
            
        
        for i in range(self.num_layers):
            if self.share_weights and i > 0:
                conv = self.layers[self.shared_layer]
            else:
                conv = self.layers[f'conv_{i}']
            if self.use_edge_weight and self.use_edge_attr:
                x = conv(
                    x, edge_index, edge_weight=edge_weight,
                    edge_attr=edge_attr
                )
            elif self.use_edge_weight:
                x = conv(x, edge_index, edge_weight=edge_weight)
            elif self.use_edge_attr:
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index)
            x = self.layers['act'](x)
            x = self.layers['dropout'](x)
            x = self.layers['batch_norm'](x)
            out['hidden_states'][:, state, :] = x
            state += 1

        out['final_state'] = self.layer_pool(out)
        out['global_state'] = self.graph_pool(out['final_state'], batch)
        return out


        