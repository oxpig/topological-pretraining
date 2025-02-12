from .mlp import MLP

import torch

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
        final_act: str = None,
    ):
        super(PredHead, self).__init__(
            input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim,
            num_layers=num_layers, dropout=dropout, batch_norm=batch_norm,
            act=act, final_act=final_act
        )
    
    @property
    def loss_fn(self):
        return torch.nn.Identity()

    def loss(self, x, y):
        preds = self(x)
        return self.loss_fn(preds, y)

class RegressionHead(PredHead):
    """
    Regression head for the model
    """
    def __init__(
        self,
        input_dim: int, 
        hidden_dim: int = None, 
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_norm: bool = False,
        act: str = 'relu',
    ):
        output_dim = 1
        super(RegressionHead, self).__init__(
            input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim,
            num_layers=num_layers, dropout=dropout, batch_norm=batch_norm,
            act=act
        )
    
    @property
    def loss_fn(self):
        return torch.nn.MSELoss()

    def loss(self, x, y):
        pred = self(x)
        return self.loss_fn(pred, y)

class BinaryHead(PredHead):
    """
    Binary classification head for the model
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
        class_weights: tuple[float] = [1.0, 1.0]
    ):
        final_act = 'sigmoid'
        super(BinaryHead, self).__init__(
            input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim,
            num_layers=num_layers, dropout=dropout, batch_norm=batch_norm,
            act=act, final_act=final_act
        )
        self.class_weights = torch.tensor(class_weights)

    @property
    def loss_fn(self):
        return torch.nn.functional.binary_cross_entropy

    def loss(self, x, y):
        pred = self(x)
        weights = torch.zeros(size=y.size())
        weights[y == 0] = self.class_weights[0]
        weights[y == 1] = self.class_weights[1]
        return self.loss_fn(pred, y, weight=weights)

class MultiClassHead(PredHead):
    """
    Multi-class classification head for the model
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
        class_weights: tuple[float] = None
    ):
        final_act = 'softmax'
        super(MultiClassHead, self).__init__(
            input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim,
            num_layers=num_layers, dropout=dropout, batch_norm=batch_norm,
            act=act, final_act=final_act
        )
        if class_weights is not None:
            assert len(class_weights) == output_dim, 'Class weights must have the same length as the output dimension.'
            self.class_weights = torch.tensor(class_weights)
        else:
            self.class_weights = torch.ones(size=(output_dim,))

    @property
    def loss_fn(self):
        return torch.nn.CrossEntropyLoss(weight=self.class_weights)

    def loss(self, x, y):
        pred = self(x)
        return self.loss_fn(pred, y)
