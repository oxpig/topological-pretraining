from topological_pretraining.nn.mlp import MLP
from topological_pretraining.nn.pred_head import BinaryHead, MultiClassHead, RegressionHead

import torch

decoders = {
    'binary': BinaryHead,
    'multiclass': MultiClassHead,
    'regression': RegressionHead
}

class AutoEncoder(torch.nn.Module):
    """
    AutoEncoder class for encoding and decoding data.
    
    Parameters:
    ----------
    input_dim : int
        The dimension of the input data.
    hidden_dim : int
        The dimension of the hidden layer.
    latent_dim : int
        The dimension of the latent space.
    encoder_layers : int, optional
        The number of layers in the encoder. Default is 1.
    decoder_layers : int, optional
        The number of layers in the decoder. Default is 1.
    dropout : float, optional
        The dropout rate. Default is 0.0.
    batch_norm : bool, optional
        Whether to use batch normalization. Default is False.
    act : str, optional
        The activation function to use. Default is 'relu'.
    decoder_type : str, optional
        The type of decoder to use. Can be 'binary', 'multiclass', or 'regression'. Default is 'regression'.
    class_weights : None, optional
        Class weights for the decoder. Default is None.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        encoder_layers: int = 1,
        decoder_layers: int = 1,
        dropout: float = 0.0,
        batch_norm: bool = False,
        act: str = 'relu',
        decoder_type: str = 'regression',
        class_weights: None = None,
    ):
        super(AutoEncoder, self).__init__()
        self.encoder = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=latent_dim,
            num_layers=encoder_layers,
            dropout=dropout,
            batch_norm=batch_norm,
            act=act,
            final_act=act,
        )
        self.decoder = decoders[decoder_type](
            input_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            num_layers=decoder_layers,
            dropout=dropout,
            batch_norm=batch_norm,
            act=act,
            class_weights=class_weights,
        )

    def forward(self, x):
        """
        Forward pass through the AutoEncoder.

        Parameters:
        ----------
        x : torch.Tensor
            Input tensor to the AutoEncoder.

        Returns:
        -------
        torch.Tensor
            Output tensor after encoding and decoding.
        """
        return self.encoder(x)

    def pred(self, x,):
        """
        Predict the output using the AutoEncoder.

        Parameters:
        ----------
        x : torch.Tensor
            Input tensor to the AutoEncoder.
        """
        enc = self(x)
        pred = self.decoder(enc)
        return pred


