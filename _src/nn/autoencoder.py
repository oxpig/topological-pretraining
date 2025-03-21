from _src.nn.mlp import MLP
from _src.nn.pred_head import BinaryHead, MultiClassHead, RegressionHead

import torch

decoders = {
    'binary': BinaryHead,
    'multiclass': MultiClassHead,
    'regression': RegressionHead
}

class AutoEncoder(torch.nn.Module):

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
        class_weight: None = None,
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
            class_weight=class_weight,
        )

    def forward(self, x):
        return self.encoder(x)

    def loss(self, x, score: bool = False):
        enc = self.encoder(x)
        pred = self.decoder(enc)
        loss = self.decoder.loss(x, pred)
        if score:
            score = self.decoder.score(x, pred)
            return loss, score
        return loss


