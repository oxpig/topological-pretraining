from ..nn.mlp import MLP

class PredHead(MLP):
    """
    Prediction head for the model
    """
    def __init__(
        self, 
    ):
        pass
        
    def forward(self, x):
        pass

    def loss(self, x, y):
        pass

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