from ..nn.mlp import MLP

class PredHead(MLP):
    """
    Prediction head for the model
    """
    def __init__(
        self,
        input_dim: int, 
        output_dim: int,
        hidden_dim: int = None, 
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_norm: bool = False,
        act: str = 'relu',
    ):
        super(PredHead, self).__init__(
            input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim,
            num_layers=num_layers, dropout=dropout, batch_norm=batch_norm,
            act=act
        )

    def loss(self, x, y):
        raise NotImplementedError

class RegressionHead(PredHead):
    """
    Regression head for the model
    """

class BinaryHead(PredHead):
    """
    Binary classification head for the model
    """

class MultiClassHead(PredHead):
    """
    Multi-class classification head for the model
    """