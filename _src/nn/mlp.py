import torch

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

class MLP(torch.nn.Module):
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
    ):
        super(MLP, self).__init__()
        if hidden_dim is None:
            hidden_dim = output_dim
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
        self.batch_norm = torch.nn.BatchNorm1d(input_dim) if batch_norm else torch.nn.Identity()
        self.act = act_fn[act]
        self.final_act = act_fn[final_act]

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.act(x)
                x = self.dropout(x)
                x = self.batch_norm(x)
        x = self.final_act(x)
        return x