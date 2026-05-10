import torch_geometric as pyg

from .gnn import BaseGNN
from .mlp import MLP


class GINLayer(pyg.nn.conv.GINConv):
	"""
	Wrapper for the GINConv layer from PyTorch Geometric.

	This layer uses a Multi-Layer Perceptron (MLP) to compute the node embeddings
	based on the aggregated neighbor features. It allows for customization of the
	MLP architecture, including input and output dimensions, number of hidden layers,
	dropout rates, batch normalization, activation functions, and initialization methods.

	Parameters:
	----------
	mlp : MLP, optional
	    An instance of the MLP class to be used for the GIN layer. If None, a new MLP
	    will be created based on the provided parameters.
	input_dim : int, optional
	    The dimension of the input features. Required if `mlp` is None.
	hidden_dim : int, optional
	    The dimension of the hidden layers in the MLP. Required if `mlp` is None.
	output_dim : int, optional
	    The dimension of the output features. Required if `mlp` is None.
	num_layers : int, optional
	    The number of hidden layers in the MLP. Default is 1.
	dropout : float, optional
	    The dropout rate applied to the MLP layers. Default is 0.0.
	batch_norm : bool, optional
	    Whether to apply batch normalization to the MLP layers. Default is False.
	act : str, optional
	    The activation function to use in the MLP. Default is 'relu'.
	eps : float, optional
	    The epsilon value for the GIN layer, which is added to the aggregated features
	    to prevent numerical instability. Default is 0.0.
	train_eps : bool, optional
	    Whether to train the epsilon value. If True, the epsilon value will be a learnable
	    parameter. Default is False.
	weight_init : str, optional
	    The initialization method for the weights of the MLP. Default is 'standard'.
	bias_init : str, optional
	    The initialization method for the biases of the MLP. Default is 'standard'.
	**kwargs : dict, optional
	    For compatibility.
	"""

	def __init__(
		self,
		mlp: MLP = None,
		input_dim: int = None,
		hidden_dim: int = None,
		output_dim: int = None,
		num_layers: int = 1,
		dropout: float = 0.0,
		batch_norm: bool = False,
		act: str = 'relu',
		eps: float = 0.0,
		train_eps: bool = False,
		weight_init: str = 'standard',
		bias_init: str = 'standard',
		**kwargs,
	):
		if mlp is None:
			mlp = MLP(
				input_dim=input_dim,
				output_dim=output_dim,
				hidden_dim=hidden_dim,
				num_layers=num_layers,
				dropout=dropout,
				batch_norm=batch_norm,
				act=act,
				weight_init=weight_init,
				bias_init=bias_init,
			)
		else:
			mlp = mlp

		super().__init__(nn=mlp, eps=eps, train_eps=train_eps)


class GIN(BaseGNN):
	"""
	Graph Isomorphism Network (GIN) model.

	This model extends the BaseGNN class and implements the GIN architecture,
	which is designed to learn graph representations by aggregating node features
	and applying a Multi-Layer Perceptron (MLP) to the aggregated features

	Parameters:
	----------
	input_dim : int, optional
	    The dimension of the input node features. Required if `mlp` is None.
	output_dim : int, optional
	    The dimension of the output node features. Required if `mlp` is None.
	**kwargs : dict, optional
	    Additional parameters for the GIN layer. See GINLayer for details.
	"""

	def _init_layer(self, mlp=None, input_dim=None, output_dim=None, **kwargs):

		if 'act' not in kwargs:
			kwargs['act'] = self.act_type
		if 'dropout' not in kwargs:
			kwargs['dropout'] = self.dropout_value
		if 'batch_norm' not in kwargs:
			kwargs['batch_norm'] = self.use_batch_norm

		if mlp is None:
			return GINLayer(
				mlp=None, input_dim=input_dim, output_dim=output_dim, **kwargs
			)
		else:
			return GINLayer(mlp=mlp, **kwargs)

	def reset_parameters(self):
		"""
		Reset parameters of the GIN layers.
		"""
		for layer in self.layers.values():
			if isinstance(layer, GINLayer):
				layer.nn.reset_parameters()
