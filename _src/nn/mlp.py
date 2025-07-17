import torch
from typing import Callable

act_fn = {
    'relu': torch.nn.ReLU(),
    'tanh': torch.nn.Tanh(),
    'sigmoid': torch.nn.Sigmoid(),
    'gelu': torch.nn.GELU(),
    'elu': torch.nn.ELU(),
    'swish': torch.nn.SiLU(),
    'hardswish': torch.nn.Hardswish(),
    'softmax': torch.nn.Softmax(dim=-1),
    None: torch.nn.Identity()
}

initializers = {
    'standard': 'standard',
    'xavier_uniform': torch.nn.init.xavier_uniform_,
    'xavier_normal': torch.nn.init.xavier_normal_,
    'normal': torch.nn.init.normal_,
    'zeros': torch.nn.init.zeros_,
    'ones': torch.nn.init.ones_,
}

class MLP(torch.nn.Module):
    """
    Multi-Layer Perceptron (MLP) model.

    This model implements a feedforward neural network with multiple layers,
    where each layer consists of a linear transformation followed by an activation function.
    
    Parameters:
    ----------
    input_dim : int
        The dimension of the input features.
    hidden_dim : int
        The dimension of the hidden layers.
    output_dim : int, optional
        The dimension of the output features. If None, it defaults to `hidden_dim`.
    num_layers : int, optional
        The number of layers in the MLP. Defaults to 1.
    dropout : float, optional
        The dropout rate applied after each layer. Defaults to 0.0.
    batch_norm : bool, optional
        Whether to apply batch normalization after each layer. Defaults to False.
    act : str, optional
        The activation function to use in the MLP. Defaults to 'relu'.
    final_act : str, optional
        The activation function to apply to the final output. If None, no activation is applied.
        Defaults to None.
    weight_init : str or Callable, optional
        The initialization method for the weights of the MLP layers.
        If a string, it must be one of the keys in `initializers`. Defaults to
        'standard', which uses the standard initialization method.
    bias_init : str or Callable, optional
        The initialization method for the biases of the MLP layers.
        If a string, it must be one of the keys in `initializers`. Defaults to
        'standard', which uses the standard initialization method.
    """
    def __init__(
        self,
        input_dim: int, 
        hidden_dim: int,
        output_dim: int = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_norm: bool = False,
        act: str = 'relu',
        final_act: str = None,
        weight_init: str|Callable = 'standard',
        bias_init: str|Callable = 'standard',
    ):
        super(MLP, self).__init__()
        if isinstance(weight_init, str):
            if weight_init not in initializers:
                raise ValueError(
                    f"Invalid weights initializer: {weight_init}. Must be callable or one of {list(initializers.keys())}."
                )
            weight_init = initializers[weight_init]
        if isinstance(bias_init, str):
            if bias_init not in initializers:
                raise ValueError(
                    f"Invalid bias initializer: {bias_init}. Must be callable or one of {list(initializers.keys())}."
                )
            bias_init = initializers[bias_init]
        
        self.weight_init = weight_init
        self.bias_init = bias_init
        if output_dim is None:
            output_dim = hidden_dim
        if num_layers == 1:
            self.layers = torch.nn.ModuleList([
                torch.nn.Linear(input_dim, output_dim)
            ])

        elif num_layers == 2:
            self.layers = torch.nn.ModuleList([
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.Linear(hidden_dim, output_dim)
            ])
        else:
            self.layers = torch.nn.ModuleList([
                torch.nn.Linear(input_dim, hidden_dim),
                *[torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 2)],
                torch.nn.Linear(hidden_dim, output_dim)
            ])
        
        self.dropout = torch.nn.Dropout(dropout)
        self.batch_norm = torch.nn.BatchNorm1d(hidden_dim) if batch_norm else torch.nn.Identity()
        self.act = act_fn[act]
        self.final_act = act_fn[final_act]
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout_value = dropout
        self.use_batch_norm = batch_norm
        self.act_type = act
        self.final_act_type = final_act

    def forward(self, x):
        """
        Forward pass of the MLP.

        Parameters:
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).
        
        Returns:
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_dim) after passing through the MLP.
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.act(x)
                x = self.dropout(x)
                x = self.batch_norm(x)
        x = self.final_act(x)
        return x
    
    def reset_parameters(self) -> None:
        """
        Reset the parameters of the MLP.
        """
        for layer in self.layers:

            if not isinstance(layer, torch.nn.Linear):
                continue
            if self.weight_init == 'standard':
                layer.reset_parameters()
            else:
                self.weight_init(layer.weight)
            if layer.bias is not None and self.bias_init != 'standard':
                self.bias_init(layer.bias)
            