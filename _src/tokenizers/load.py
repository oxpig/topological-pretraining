from . import (
    BaseTokenizer,
    AtomGraphTokenizer, MorganGraphTokenizer,
    ECFP, FCFP, PDV, SNS
)
import torch

all_tokenizers = {
    'AtomGraphTokenizer': AtomGraphTokenizer,
    'MorganGraphTokenizer': MorganGraphTokenizer,
    'ECFP': ECFP,
    'FCFP': FCFP,
    'PDV': PDV,
    'SNS': SNS
}


def read_from_dict(parameters: dict) -> BaseTokenizer:
    """
    Load tokenizer from a dictionary
    """
    if 'name' not in parameters:
        raise ValueError('Tokenizer name not found in parameters.')
    tokenizer = parameters.pop('name')
    tokenizer = all_tokenizers[tokenizer](**parameters)
    tokenizer.is_fitted_ = parameters['is_fitted_']
    return tokenizer


def load_tokenizer(path: str, parameters: bool = True) -> BaseTokenizer:
    """
    Load a tokenizer from a file.
    """
    tokenizer = torch.load(path, weights_only=parameters)
    if parameters:
        return read_from_dict(tokenizer)
    else:
        return tokenizer