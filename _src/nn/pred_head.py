from .mlp import MLP

import numpy as np
from sklearn.metrics import average_precision_score
import torch
from torcheval.metrics.functional import binary_auprc

import warnings


class PredHead(MLP):
    """
    Prediction head for a model.

    This class extends the MLP class to create a prediction head that can be used
    for various tasks such as regression, binary classification, and multi-class
    classification. It provides methods for computing loss and score based on the
    predictions and ground truth labels.

    Parameters:
    ----------
    input_dim : int
        The dimension of the input features.
    output_dim : int
        The dimension of the output features.
    hidden_dim : int, optional
        The dimension of the hidden layers. If None, it defaults to `input_dim`.
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
        """
        This property should be overridden in subclasses to provide the appropriate loss function.
        """
        return torch.nn.Identity()

    def loss(self, y, pred, mask=None):
        """
        Compute the loss between the predicted values and the ground truth labels.
        
        Parameters:
        ----------
        y : torch.Tensor
            The ground truth labels.
        pred : torch.Tensor
            The predicted values from the model.
        mask : torch.Tensor, optional
            A mask to apply to the loss values. If provided, the loss will be computed only
            for the elements where the mask is True. Defaults to None.

        Returns:
        -------
        torch.Tensor
            The computed loss values. If a mask is provided, the loss will be averaged over
            the masked elements. If no mask is provided, the loss will be averaged over all elements.
        """
        dtype = pred.dtype
        y = y.type(dtype)
        loss_vals = self.loss_fn(pred, y)

        if mask is not None:
            mask.type(dtype)
            loss_vals = loss_vals * mask
        loss_vals = loss_vals.mean(dim=0)
        return loss_vals
    
    def score(self, y, pred, mask=None):
        """
        Caculate a custom score for the predictions. Override this method in subclasses
        to implement specific scoring logic.
        """
        return -np.inf
    
    def set_class_weight(self, class_weights: torch.Tensor):
        raise NotImplementedError(
            "This method should be overridden in subclasses to set class weights for the loss function."
        )

class RegressionHead(PredHead):
    """
    Regression head for a model.

    This class extends the PredHead class to create a regression head that can be used
    for regression tasks. It provides a specific loss function for regression and can be
    used to compute the mean squared error between the predicted values and the ground truth labels.

    Parameters:
    ----------
    input_dim : int
        The dimension of the input features.
    hidden_dim : int, optional
        The dimension of the hidden layers. If None, it defaults to `input_dim`.
    output_dim : int, optional
        The dimension of the output features. Defaults to 1.
    num_layers : int, optional
        The number of layers in the MLP. Defaults to 1.
    dropout : float, optional
        The dropout rate applied after each layer. Defaults to 0.0.
    batch_norm : bool, optional
        Whether to apply batch normalization after each layer. Defaults to False.
    act : str, optional
        The activation function to use in the MLP. Defaults to 'relu'.
    **kwargs : dict, optional
    """
    def __init__(
        self,
        input_dim: int, 
        hidden_dim: int = None, 
        output_dim: int = 1,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_norm: bool = False,
        act: str = 'relu',
        **kwargs,
    ):
        super(RegressionHead, self).__init__(
            input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim,
            num_layers=num_layers, dropout=dropout, batch_norm=batch_norm,
            act=act
        )
    
    @property
    def loss_fn(self):
        """
        Mean Squared Error loss function for regression tasks.

        Returns:
        -------
        torch.nn.MSELoss
            The mean squared error loss function with 'mean' reduction.
        """
        return torch.nn.MSELoss(reduction='mean')

class BinaryHead(PredHead):
    """
    Binary classification head for a model.
    This class extends the PredHead class to create a binary classification head that can be used
    for binary classification tasks. It provides a specific loss function for binary classification
    and can be used to compute the binary cross-entropy loss between the predicted probabilities
    and the ground truth labels.

    Parameters:
    ----------
    input_dim : int
        The dimension of the input features.
    output_dim : int
        The dimension of the output features. Defaults to 1.
    hidden_dim : int, optional
        The dimension of the hidden layers. If None, it defaults to `input_dim`.
    num_layers : int, optional
        The number of layers in the MLP. Defaults to 1.
    dropout : float, optional
        The dropout rate applied after each layer. Defaults to 0.0.
    batch_norm : bool, optional
        Whether to apply batch normalization after each layer. Defaults to False.
    act : str, optional
        The activation function to use in the MLP. Defaults to 'relu'.
    class_weights : torch.Tensor, optional
        A tensor containing the class weights for the binary classification task.
        If provided, it will be used to weight the loss function. Defaults to None.
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
        self.set_class_weight(class_weights)

    def set_class_weight(self, class_weights = None):
        """
        Set class weights for the binary classification task.
        This method sets the class weights for the binary classification task.

        Parameters:
        ----------
        class_weights : torch.Tensor, optional
            A tensor containing the class weights for the binary classification task.
            If None, no class weights will be set. Defaults to None.
        """
        if class_weights is None:
            self.class_weights = None
            return
        if class_weights.dim() == 2:
            class_weights = class_weights.unsqueeze(1)
        assert class_weights.dim() == 3, 'Class weights must be a 3D tensor.'
        assert class_weights.size(0) == 2, 'Class weights must have a value for each class at dim 0, 0 and 1.'
        assert class_weights.size(1) == 1, 'Class weights must have a dimension at dim 1 of length 1 for repeats.'
        assert class_weights.size(-1) == self.output_dim, 'Class weights must have a values for each task at dim 2.'
        self.class_weights = class_weights

    @property
    def loss_fn(self):
        """
        Binary Cross-Entropy loss function for binary classification tasks.

        Returns:
        -------
        torch.nn.functional.binary_cross_entropy
        """
        return torch.nn.functional.binary_cross_entropy
    
    def score(self, y, pred, mask=None):
        """
        Calculate the binary average precision score for the predictions.

        Note: will output nan if there are no positive samples a batch.

        Parameters:
        ----------
        y : torch.Tensor
            The ground truth labels for the binary classification task.
        pred : torch.Tensor
            The predicted probabilities from the model.
        mask : torch.Tensor, optional
            A mask to apply to the predictions and ground truth labels. If provided, the score
            will be computed only for the elements where the mask is True. Defaults to None.

        Returns:
        -------
        torch.Tensor
            The computed binary average precision score. If a mask is provided, the score will be
            averaged over the masked elements. If no mask is provided, the score will be averaged
            over all elements.
        """
        y = y.type(pred.dtype)
        if mask is not None:
            y = y[mask]
            pred = pred[mask]
        y = y.transpose(0, 1)
        pred = pred.transpose(0, 1)
        return binary_auprc(input=pred, target=y, num_tasks=self.output_dim).mean()

    def loss(self, y, pred, mask=None):
        """
        Calculate the binary cross-entropy loss for the predictions.

        Parameters:
        ----------
        y : torch.Tensor
            The ground truth labels for the binary classification task.
        pred : torch.Tensor
            The predicted probabilities from the model.
        mask : torch.Tensor, optional
            A mask to apply to the loss values. If provided, the loss will be computed only
            for the elements where the mask is True. Defaults to None.

        Returns:
        -------
        torch.Tensor
            The computed loss value. If a mask is provided, the loss will be averaged over
            the masked elements. If no mask is provided, the loss will be averaged over all elements.
        """
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
    Multi-class classification head for the model.

    
    Parameters:
    ----------
    input_dim : int
        The dimension of the input features.
    output_dim : int
        The number of classes for multi-class classification.
    hidden_dim : int, optional
        The dimension of the hidden layers. If None, it defaults to `input_dim`.
    num_layers : int, optional
        The number of layers in the MLP. Defaults to 1.
    dropout : float, optional
        The dropout rate applied after each layer. Defaults to 0.0.
    batch_norm : bool, optional
        Whether to apply batch normalization after each layer. Defaults to False.
    act : str, optional
        The activation function to use in the MLP. Defaults to 'relu'.
    class_weights : tuple[float], optional
        A tuple containing the class weights for each class in the multi-class classification task.
        If provided, it will be used to weight the loss function. Defaults to None.
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
        self.set_class_weight(class_weights)

    def set_class_weight(self, class_weights = None):
        """
        Set class weights for the multi-class classification task.

        Parameters:
        ----------
        class_weights : tuple[float], optional
            A tuple containing the class weights for each class in the multi-class classification task.
            If None, no class weights will be set. Defaults to None.
        """
        if class_weights is not None:
            assert len(class_weights) == self.output_dim, 'Class weights must have the same length as the output dimension.'
            self.class_weights = torch.tensor(class_weights)
        else:
            self.class_weights = torch.ones(size=(self.output_dim,))

    @property
    def loss_fn(self):
        """
        Cross-Entropy loss function for multi-class classification tasks.

        Returns:
        -------
        torch.nn.CrossEntropyLoss
            The cross-entropy loss function with 'none' reduction, which means the loss will be computed
            for each element in the batch without averaging.
        """
        return torch.nn.CrossEntropyLoss(weight=self.class_weights, reduction='none')

class MultiTaskLoss(torch.nn.Module):
    """
    https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf
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
