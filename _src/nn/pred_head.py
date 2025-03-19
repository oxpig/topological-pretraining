from .mlp import MLP

import numpy as np
from sklearn.metrics import average_precision_score
import torch
from torcheval.metrics import BinaryAUPRC, MultilabelAUPRC

import warnings


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

    def loss(self, y, pred, mask=None):
        dtype = pred.dtype
        y = y.type(dtype)
        loss_vals = self.loss_fn(pred, y)

        if mask is not None:
            mask.type(dtype)
            loss_vals = loss_vals * mask
        loss_vals = loss_vals.mean(dim=0)
        return loss_vals
    
    def score(self, y, pred, mask=None):
        return -np.inf

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
        **kwargs,
    ):
        output_dim = 1
        super(RegressionHead, self).__init__(
            input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim,
            num_layers=num_layers, dropout=dropout, batch_norm=batch_norm,
            act=act
        )
    
    @property
    def loss_fn(self):
        return torch.nn.MSELoss(reduction='mean')

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
        class_weights: torch.Tensor = None,
        **kwargs,
    ):
        final_act = 'sigmoid'
        super(BinaryHead, self).__init__(
            input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim,
            num_layers=num_layers, dropout=dropout, batch_norm=batch_norm,
            act=act, final_act=final_act
        )
        if class_weights is None:
            self.class_weights = None
        else:
            if class_weights.dim() == 2:
                class_weights = class_weights.unsqueeze(1)
            assert class_weights.dim() == 3, 'Class weights must be a 3D tensor.'
            assert class_weights.size(0) == 2, 'Class weights must have a value for each class at dim 0, 0 and 1.'
            assert class_weights.size(1) == 1, 'Class weights must have a dimension at dim 1 of length 1 for repeats.'
            assert class_weights.size(-1) == output_dim, 'Class weights must have a values for each task at dim 2.'
            self.class_weights = class_weights
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.output_dim == 1:
            self.score_fn = BinaryAUPRC(num_tasks=1, device=self.device)
        else:
            self.score_fn = MultilabelAUPRC(num_tasks=self.output_dim, device=self.device,)
    

    @property
    def loss_fn(self):
        return torch.nn.functional.binary_cross_entropy

    def score(self, y, pred, mask=None):

        y = y.type(pred.dtype)
        if mask is not None:
            y = y[mask]
            pred = pred[mask]
        y = y.transpose(0, 1)
        pred = pred.transpose(0, 1)
        return self.score_fn(pred, y)

    def loss(self, y, pred, mask=None):
        y = y.type(pred.dtype)
        if self.class_weights is None:
            return self.loss_fn(pred, y,)
        weights = torch.zeros(size=y.size(), dtype=pred.dtype).to(y.device)
        class_weights = self.class_weights.repeat(1, weights.size(0), 1)
        class_weights = class_weights.type(pred.dtype)
        class_weights = class_weights.to(y.device)
        weights[y == 0] = class_weights[0, y == 0]
        weights[y == 1] = class_weights[1, y == 1]
        loss_vals = self.loss_fn(pred, y, weight=weights, reduction='none')
        loss_vals = loss_vals.mean(dim=1)
        if mask is not None:
            loss_vals = loss_vals * mask
            loss_vals = loss_vals.sum(dim=0) / mask.sum()
        else:
            loss_vals = loss_vals.mean(dim=0)
        return loss_vals
    

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
        class_weights: tuple[float] = None,
        **kwargs,
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
        return torch.nn.CrossEntropyLoss(weight=self.class_weights, reduction='none')

class MultiTaskLoss(torch.nn.Module):
    """
    From https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf

    """
    def __init__(self, is_regression: torch.BoolTensor):
        super(MultiTaskLoss, self).__init__()
        self.is_regression = torch.ones_like(is_regression)
        self.is_regression[is_regression] = 2
        self.num_heads = is_regression.size(0)
        self.sigmas = torch.nn.Parameter(torch.zeros(self.num_heads))

    def forward(self, losses: torch.Tensor):
        assert losses.size(0) == self.num_heads, 'Number of losses must match the number of heads.'
        losses = ((1 / (self.is_regression * self.sigmas**2)) * losses + torch.log(self.sigmas))
        return losses.sum()
