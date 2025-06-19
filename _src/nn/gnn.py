from .mlp import act_fn

import torch
import torch_geometric as pyg
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
        layer_pool_type: slice|int|Literal[
            'last', 'sum', 'mean',
            'max', 'concat'
        ] = 'concat',
        graph_pool_type: Literal[
            None, 'sum', 'mean',
            'max', 'global_node'
        ] = 'max',
        gnn_kwargs: dict = {},
        share_weights: bool = False,
        device: str = 'cpu',
        seed: int = 42,
    ):
        super(BaseGNN, self).__init__()
        torch.manual_seed(seed)
        self.seed = seed
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

        else:
            for i in range(layer_count, num_layers):
                self.layers[f'conv_{i}'] = self._init_layer(
                    input_dim=hidden_dim, hidden_dim=hidden_dim, **gnn_kwargs
                )

        self.out_shape = self.cal_out_shape()

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
        elif isinstance(self.layer_pool_type, int):
            out = hidden_states[:, self.layer_pool_type, :]
        elif isinstance(self.layer_pool_type, slice):
            out = hidden_states[:, self.layer_pool_type, :]
            out = out.view(out.size(0), -1)
        else:
            raise ValueError(
                f'Invalid layer pooling method: {self.layer_pool_type}\n'\
                'Valid options are: last, sum, mean, max, concat.'
            )
        return out
    
    def graph_pool(
        self, final_state: torch.Tensor, batch: torch.Tensor,
        global_idx: torch.Tensor = None
    ):
        if self.graph_pool_type is None:
            return None
        elif self.graph_pool_type == 'sum':
            return pyg.nn.pool.global_add_pool(final_state, batch)
        elif self.graph_pool_type == 'mean':
            return pyg.nn.pool.global_mean_pool(final_state, batch)
        elif self.graph_pool_type == 'max':
            return pyg.nn.pool.global_max_pool(final_state, batch,)
        elif self.graph_pool_type == 'global_node':
            assert global_idx is not None, 'Global node index must be provided for global node pooling.'
            return final_state[global_idx]
        else:
            raise ValueError(
                f'Invalid graph pooling method: {self.graph_pool_type}\n'\
                'Valid options are: sum, mean, max.'
            )
        
    def prepare_out(self, x: torch.Tensor):

        out = {}
        num_hidden_states = self.num_hidden_states
        out['hidden_states'] = torch.zeros(
            (x.size(0), num_hidden_states, self.hidden_dim)
        ).to(x.device)
        out['state'] = 0
        return out
    
    def embed_nodes(
        self, x: torch.Tensor, embedded: torch.Tensor = torch.tensor([False]), batch: torch.Tensor = None
    ):
        if embedded.all():
            return x, embedded
        elif embedded.any():
            raise ValueError('Some graphs in batch are already embedded.')
        else:
            if batch is None:
                embedded = torch.tensor([True], dtype=torch.bool).to(x.device)
            else:
                embedded = torch.full((batch.max()+1, 1), True).to(x.device)
            return self.layers['node_embedding'](x), embedded
        
    def embed_graph_nodes(self, graph: pyg.data.Data, keep_tokens=False):
        if self.node_vocab_size is None:
            Warning(
                'Node embedding is not defined. Returning original graph.'
            )
            return graph
        x, embedded = graph.get("x", False), graph.get("embedded", torch.tensor([False])).to(graph.x.device)
        if x is False:
            Warning(
                'Graph does not have node features. Returning original graph.'
            )
            return graph
        if keep_tokens:
            graph.tokens = graph.x.clone()
        graph.x, graph.embedded = self.embed_nodes(x, embedded)
        return graph
        
    def prepare_embedding(
        self, out: dict, x: torch.Tensor, embedded: torch.Tensor,
        batch: torch.Tensor = None
    ):
        if self.node_vocab_size is not None:
            x, _ = self.embed_nodes(x, embedded, batch)
            if self.node_embedding_dim == self.hidden_dim:
                out['hidden_states'][:, :x.size(1), :] = x
                out['node_embedding'] = None
                out['state'] += x.size(1)
            else:
                out['node_embedding'] = x
            x = x.view(x.size(0), -1)
        else:
            out['node_embedding'] = None
        return out, x
        
    def convolutions(
        self, out, x, edge_index, edge_weight=None, edge_attr=None
    ):
        for i in range(self.num_layers):
            x = self.layers['dropout'](x)
            if i != 0 and self.use_batch_norm:
                x = self.layers['batch_norm'](x)
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
            out['hidden_states'][:, out['state'], :] = x
            out['state'] += 1
        return out
    
    def pooling(self, out: dict, batch: torch.Tensor, global_idx):
        out['final_state'] = self.layer_pool(out)
        out['global_state'] = self.graph_pool(
            out['final_state'], batch, global_idx
        )
        return out
    
    def record_node_embedding(self, out: dict):
        if out['node_embedding'] is None:
            out['node_embedding'] = out['hidden_states'][:, :self.input_dim, :]
        return out
    
    def forward(
        self, x, edge_index,
        edge_weight = None, edge_attr = None, batch = None,
        global_idx = None, embedded = torch.tensor([False]), **kwargs
    ):
        if x is None: return None
        embedded = embedded.to(x.device)
        out = self.prepare_out(x)
        out, x = self.prepare_embedding(out, x, embedded, batch)
        out = self.convolutions(
            out=out, x=x, edge_index=edge_index,
            edge_weight=edge_weight, edge_attr=edge_attr
        )
        out = self.pooling(out=out, batch=batch, global_idx=global_idx)
        out = self.record_node_embedding(out)
        
        return out
    
    def parse_example_graph(self,):
        if self.node_embedding_dim is not None:
            x_dtype = torch.long
        else:
            x_dtype = torch.float
        
        example_graph = pyg.data.Data(
            x=torch.zeros((1, self.input_dim), dtype=x_dtype),
            edge_index=torch.tensor([[0],[0]], dtype=torch.long),
            edge_attr=torch.zeros((1, self.input_dim), dtype=torch.long),
            batch=torch.tensor([0], dtype=torch.long),
            global_idx=torch.tensor([0], dtype=torch.long),
        )
        example_graph = example_graph.to(next(self.parameters()).device)
        with torch.no_grad():
            self.eval()
            out = self(**example_graph)
        return out

    def cal_out_shape(self):
        out = self.parse_example_graph()
        return out['global_state'].size(1)

