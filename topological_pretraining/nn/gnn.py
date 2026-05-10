from typing import Literal

import torch
import torch_geometric as pyg

from .mlp import act_fn


class BaseGNN(torch.nn.Module):
	"""
	Base class for Graph Neural Networks (GNNs).

	This class provides a framework for building GNNs with configurable parameters
	such as input and hidden dimensions, dropout rates, activation functions, and
	pooling methods. It supports both node and graph-level embeddings and allows for
	customization of the GNN layers through the `gnn_kwargs` parameter.

	Parameters:
	----------
	input_dim : int
	    The dimension of the input features.
	hidden_dim : int
	    The dimension of the hidden layers in the GNN.
	node_embedding : tuple[int, int], optional
	    A tuple specifying the vocabulary size and embedding dimension for node features.
	    If None, no node embedding is used.
	padding_idx : int, optional
	    The index to use for padding in the node embedding layer. Default is None.
	num_layers : int, optional
	    The number of GNN layers. Default is 1.
	dropout : float, optional
	    The dropout rate applied to the GNN layers. Default is 0.0.
	batch_norm : bool, optional
	    Whether to apply batch normalization to the GNN layers. Default is False.
	act : str, optional
	    The activation function to use in the GNN layers. Default is 'relu'.
	layer_pool_type : slice|int|Literal['last', 'sum', 'mean',
	    'max', 'concat'], optional
	    The pooling method to apply to the hidden states of the GNN layers.
	    Options include 'last', 'sum', 'mean', 'max', 'concat', or a specific layer index or slice.
	    Default is 'concat'.
	graph_pool_type : Literal[None, 'sum', 'mean', 'max', 'global_node'], optional
	    The pooling method to apply to the final node embeddings to obtain a graph-level embedding.
	    Default is 'max'.
	gnn_kwargs : dict, optional
	    Additional keyword arguments to pass to the GNN layer initialization.
	    This can include parameters like activation function, dropout rate, and batch normalization.
	share_weights : bool, optional
	    Whether to share weights across GNN layers. If True, all layers will use the same
	    weights for the GNN convolution. Default is False.
	device : str, optional
	    For compatibility.
	"""

	use_edge_weight: bool = False
	use_edge_attr: bool = False

	def __init__(
		self,
		input_dim: int,
		hidden_dim: int,
		node_embedding: tuple[int, int] = None,
		padding_idx: int = None,
		num_layers: int = 1,
		dropout: float = 0.0,
		batch_norm: bool = False,
		act: str = 'relu',
		layer_pool_type: slice
		| int
		| Literal['last', 'sum', 'mean', 'max', 'concat'] = 'concat',
		graph_pool_type: Literal[None, 'sum', 'mean', 'max', 'global_node'] = 'max',
		gnn_kwargs: dict = None,
		share_weights: bool = False,
		device: str = 'cpu',
		seed: int = 42,
	):
		super().__init__()
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
		self.gnn_kwargs = gnn_kwargs or {}
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
		self.layers = torch.nn.ModuleDict(
			{
				'act': act_fn[act],
				'dropout': torch.nn.Dropout(dropout),
				'batch_norm': torch.nn.BatchNorm1d(hidden_dim)
				if batch_norm
				else torch.nn.Identity(),
			}
		)
		self.num_hidden_states = self.num_layers
		if self.node_vocab_size is not None:
			self.layers['node_embedding'] = torch.nn.Embedding(
				self.node_vocab_size, self.node_embedding_dim, padding_idx=padding_idx
			)
			node_tokens = input_dim
			input_dim = self.node_embedding_dim * node_tokens
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
			for i in range(layer_count + 1, num_layers):
				self.layers[f'conv_{i}'] = self._init_layer(
					mlp=self.layers[f'conv_{layer_count}'].nn, **gnn_kwargs
				)

		else:
			for i in range(layer_count, num_layers):
				self.layers[f'conv_{i}'] = self._init_layer(
					input_dim=hidden_dim, hidden_dim=hidden_dim, **gnn_kwargs
				)

		self.out_shape = self.cal_out_shape()

	def _init_layer(self, input_dim, output_dim, **kwargs):
		"""
		Initialize a GNN layer with the specified input and output dimensions.
		This method should be overridden in subclasses to define the specific GNN layer type.

		Parameters:
		----------
		input_dim : int
		    The dimension of the input features for the GNN layer.
		output_dim : int
		    The dimension of the output features for the GNN layer.
		**kwargs : dict, optional
		    Additional parameters for the GNN layer.
		"""
		raise NotImplementedError

	def layer_pool(self, out: dict):
		"""
		Pool the hidden states of the GNN layers based on the specified pooling method.
		This method aggregates the hidden states of the GNN layers into a single output tensor.
		This can include initial node embeddings if they are defined as learnable parameters,
		and they are compatible with the pooling type (e.g., `concat`).

		Parameters:
		----------
		out : dict
		    A dictionary containing the hidden states of the GNN layers.

		Returns:
		-------
		torch.Tensor
		    The pooled output tensor based on the specified pooling method.
		"""
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
				f'Invalid layer pooling method: {self.layer_pool_type}\n'
				'Valid options are: last, sum, mean, max, concat.'
			)
		return out

	def graph_pool(
		self,
		final_state: torch.Tensor,
		batch: torch.Tensor,
		global_idx: torch.Tensor = None,
	):
		"""
		Pool the final node embeddings to obtain a graph-level embedding.
		This method aggregates the final node embeddings based on the specified graph pooling method.

		Parameters:
		----------
		final_state : torch.Tensor
		    The final node embeddings after the GNN layers.
		batch : torch.Tensor
		    A tensor indicating the batch indices for each node in the graph.
		global_idx : torch.Tensor, optional
		    A tensor indicating the global node index for pooling. Required if `graph_pool_type` is 'global_node'.

		Returns:
		-------
		torch.Tensor
		    The pooled graph-level embedding based on the specified graph pooling method.
		"""
		if self.graph_pool_type is None:
			return None
		elif self.graph_pool_type == 'sum':
			return pyg.nn.pool.global_add_pool(final_state, batch)
		elif self.graph_pool_type == 'mean':
			return pyg.nn.pool.global_mean_pool(final_state, batch)
		elif self.graph_pool_type == 'max':
			return pyg.nn.pool.global_max_pool(
				final_state,
				batch,
			)
		elif self.graph_pool_type == 'global_node':
			if global_idx is None:
				raise ValueError(
					'Global node index must be provided for global node pooling.'
				)
			return final_state[global_idx]
		else:
			raise ValueError(
				f'Invalid graph pooling method: {self.graph_pool_type}\n'
				'Valid options are: sum, mean, max.'
			)

	def prepare_out(self, x: torch.Tensor):
		"""
		Prepare the output dictionary for the GNN layers.
		This method initializes the output dictionary with the hidden states and state index.

		Parameters:
		----------
		x : torch.Tensor
		    The input tensor containing node features.

		Returns:
		-------
		dict
		    A dictionary containing the initial hidden states and state index.
		"""
		out = {}
		num_hidden_states = self.num_hidden_states
		out['hidden_states'] = torch.zeros(
			(x.size(0), num_hidden_states, self.hidden_dim)
		).to(x.device)
		out['state'] = 0
		return out

	def embed_nodes(
		self,
		x: torch.Tensor,
		embedded: torch.Tensor = None,
		batch: torch.Tensor = None,
	):
		"""
		Embed the node features using the node embedding layer if defined.
		This method checks if the node features are already embedded and returns the
		embedded features along with a boolean tensor indicating whether the nodes are embedded.
		If nodes are already embedded, it returns the features as is.

		Parameters:
		----------
		x : torch.Tensor
		    The input tensor containing node tokens or features.
		embedded : torch.Tensor, optional
		    A boolean tensor indicating whether the nodes are already embedded.
		    Default is a tensor with a single False value.
		batch : torch.Tensor, optional
		    A tensor indicating the batch indices for each node in the graph.
		"""
		embedded = embedded or torch.tensor([False])
		if embedded.all():
			return x, embedded
		elif embedded.any():
			raise ValueError('Some graphs in batch are already embedded.')
		else:
			if batch is None:
				embedded = torch.tensor([True], dtype=torch.bool).to(x.device)
			else:
				embedded = torch.full((batch.max() + 1, 1), True).to(x.device)
			return self.layers['node_embedding'](x), embedded

	def embed_graph_nodes(self, graph: pyg.data.Data, keep_tokens=False):
		"""
		Embed the nodes of a graph using the node embedding layer if defined.
		This is for initializing the node features of a graph without passing through the GNN layers.

		Parameters:
		----------
		graph : pyg.data.Data
		    The graph data object containing node tokens and other attributes.
		keep_tokens : bool, optional
		    Whether to keep the original node tokens in the graph. If True, the original node tokens
		    will be stored in `graph.tokens`. Default is False.

		Returns:
		-------
		pyg.data.Data
		    The graph data object with embedded node features and an updated `embedded` attribute.
		    If `keep_tokens` is True, the original node tokens will be stored in `graph.tokens`.
		    If the model does not have an embedding layer for nodes, it returns the original graph.
		    If the graph does not have node tokens, it returns the original graph with a warning.
		"""
		if self.node_vocab_size is None:
			Warning('Node embedding is not defined. Returning original graph.')
			return graph
		x, embedded = (
			graph.get('x', False),
			graph.get('embedded', torch.tensor([False])).to(graph.x.device),
		)
		if x is False:
			Warning('Graph does not have node features. Returning original graph.')
			return graph
		if keep_tokens:
			graph.tokens = graph.x.clone()
		graph.x, graph.embedded = self.embed_nodes(x, embedded)
		return graph

	def prepare_embedding(
		self,
		out: dict,
		x: torch.Tensor,
		embedded: torch.Tensor,
		batch: torch.Tensor = None,
	):
		"""
		Prepare the node embeddings for the GNN layers.

		This method checks if the node features are already embedded and returns the
		embedded features along with the updated output dictionary. If the node features
		are not embedded, it applies the node embedding layer to the input tensor `x`.

		Parameters:
		----------
		out : dict
		    The output dictionary containing the hidden states and state index.
		x : torch.Tensor
		    The input tensor containing node tokens.

		Returns:
		-------
		tuple[dict, torch.Tensor]
		    A tuple containing the updated output dictionary and the embedded node features.
		    If the node embedding layer is defined, the embedded features are reshaped to match
		    the expected input dimensions for the GNN layers.
		"""
		if self.node_vocab_size is not None:
			x, _ = self.embed_nodes(x, embedded, batch)
			if self.node_embedding_dim == self.hidden_dim:
				out['hidden_states'][:, : x.size(1), :] = x
				out['node_embedding'] = None
				out['state'] += x.size(1)
			else:
				out['node_embedding'] = x
			x = x.view(x.size(0), -1)
		else:
			out['node_embedding'] = None
		return out, x

	def convolutions(self, out, x, edge_index, edge_weight=None, edge_attr=None):
		"""
		Perform the GNN convolutions on the input features.
		This method iterates through the GNN layers, applying dropout, batch normalization,
		and the GNN convolution operations. It updates the output dictionary with the hidden states
		after each layer.

		Parameters:
		----------
		out : dict
		    The output dictionary containing the hidden states and state index.
		x : torch.Tensor
		    The input tensor containing node features.
		edge_index : torch.Tensor
		    The edge index tensor representing the graph structure.
		edge_weight : torch.Tensor, optional
		    The edge weight tensor for the graph edges. Default is None.
		edge_attr : torch.Tensor, optional
		    The edge attribute tensor for the graph edges. Default is None.

		Returns:
		-------
		dict
		    The updated output dictionary containing the hidden states after each GNN layer.
		    The hidden states are stored in `out['hidden_states']`, and the state index is
		    updated after each layer.
		    The hidden states are reshaped to match the expected dimensions for the GNN layers.
		"""
		for i in range(self.num_layers):
			x = self.layers['dropout'](x)
			if i != 0 and self.use_batch_norm:
				x = self.layers['batch_norm'](x)
			conv = self.layers[f'conv_{i}']
			if self.use_edge_weight and self.use_edge_attr:
				x = conv(x, edge_index, edge_weight=edge_weight, edge_attr=edge_attr)
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
		"""
		Perform pooling on the hidden states to obtain the final node embeddings
		and the graph-level embedding. This method applies the specified layer pooling method
		to the hidden states and then applies the graph pooling method to obtain the global state.

		Parameters:
		----------
		out : dict
		    The output dictionary containing the hidden states and state index.
		batch : torch.Tensor
		    A tensor indicating the batch indices for each node in the graph.
		global_idx : torch.Tensor, optional
		    A tensor indicating the global node index for pooling. Required if `graph_pool_type` is 'global_node'.
		    Otherwise, it can be None.

		Returns:
		-------
		dict
		    The updated output dictionary containing the final node embeddings and the graph-level embedding.
		"""
		out['final_state'] = self.layer_pool(out)
		out['global_state'] = self.graph_pool(out['final_state'], batch, global_idx)
		return out

	def record_node_embedding(self, out: dict):
		"""
		Record the node embeddings in the output dictionary.
		This method checks if the node embedding layer is defined and, if so,
		assigns the node embeddings to `out['node_embedding']`. If the node embedding layer
		is not defined, it assigns the hidden states corresponding to the input dimension
		to `out['node_embedding']`.

		Parameters:
		----------
		out : dict
		    The output dictionary containing the hidden states and state index.

		Returns:
		-------
		dict
		    The updated output dictionary with the node embeddings recorded.
		    If the node embedding layer is defined, the node embeddings are stored in `out['node_embedding']`.
		    If the node embedding layer is not defined, the hidden states corresponding to the input
		    dimension are stored in `out['node_embedding']`.
		"""
		if out['node_embedding'] is None:
			out['node_embedding'] = out['hidden_states'][:, : self.input_dim, :]
		return out

	def forward(
		self,
		x,
		edge_index,
		edge_weight=None,
		edge_attr=None,
		batch=None,
		global_idx=None,
		embedded=None,
		**kwargs,
	):
		"""
		Forward pass of the GNN model.
		This method processes the input features through the GNN layers, applies pooling,
		and returns the final node embeddings and the graph-level embedding.

		Parameters:
		----------
		x : torch.Tensor
		    The input tensor containing node features or tokens.
		edge_index : torch.Tensor
		    The edge index tensor representing the graph structure.
		edge_weight : torch.Tensor, optional
		    The edge weight tensor for the graph edges. Default is None.
		edge_attr : torch.Tensor, optional
		    The edge attribute tensor for the graph edges. Default is None.
		batch : torch.Tensor, optional
		    A tensor indicating the batch indices for each node in the graph.
		    Default is None.
		global_idx : torch.Tensor, optional
		    A tensor indicating the global node index for pooling. Required if `graph_pool_type` is 'global_node'.
		    Default is None.
		embedded : torch.Tensor, optional
		    A boolean tensor indicating whether the nodes are already embedded.
		    Default is a tensor with a single False value.
		**kwargs : dict, optional
		    Additional keyword arguments for compatibility.
		"""
		if x is None:
			return None
		embedded = embedded or torch.tensor([False])
		embedded = embedded.to(x.device)
		out = self.prepare_out(x)
		out, x = self.prepare_embedding(out, x, embedded, batch)
		out = self.convolutions(
			out=out,
			x=x,
			edge_index=edge_index,
			edge_weight=edge_weight,
			edge_attr=edge_attr,
		)
		out = self.pooling(out=out, batch=batch, global_idx=global_idx)
		out = self.record_node_embedding(out)

		return out

	def parse_example_graph(
		self,
	):
		"""
		Parse an example graph to test the model's forward pass.
		This method creates a dummy graph with the specified input dimension and
		returns the output of the model's forward pass on this graph.

		Returns:
		-------
		dict
		    The output dictionary containing the final node embeddings and the graph-level embedding.
		"""
		x_dtype = torch.long if self.node_embedding_dim is not None else torch.float

		example_graph = pyg.data.Data(
			x=torch.zeros((1, self.input_dim), dtype=x_dtype),
			edge_index=torch.tensor([[0], [0]], dtype=torch.long),
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
		"""
		Calculate the output shape of the GNN model.
		This method parses an example graph and returns the size of the global state,
		which represents the output dimension of the model.

		Returns:
		-------
		int
		    The size of the global state, which is the output dimension of the GNN model.
		"""
		out = self.parse_example_graph()
		return out['global_state'].size(1)
