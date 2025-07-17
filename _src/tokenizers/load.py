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
    Load tokenizer from a dictionary.

    Parameters:
    ----------
    parameters : dict
        A dictionary containing the tokenizer's parameters. The dictionary must include
        a 'name' key that specifies the type of tokenizer to instantiate, along with any
        additional parameters required for that tokenizer.

    Returns:
    -------
    BaseTokenizer
        An instance of the specified tokenizer class, initialized with the provided parameters.

    Raises:
    ------
    ValueError
        If the 'name' key is not found in the parameters dictionary.
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

    Parameters:
    ----------
    path : str
        The file path to the saved tokenizer. The file should be a PyTorch model file.
    parameters : bool, optional
        If True, the function will return a dictionary of parameters instead of the tokenizer object.
        Defaults to True.

    Returns:
    -------
    BaseTokenizer or dict
        If `parameters` is True, loads a saved dictionary and initializes a tokenizer from it.
        If `parameters` is False, returns the tokenizer object directly.
    """
    tokenizer = torch.load(path, weights_only=parameters)
    if parameters:
        return read_from_dict(tokenizer)
    else:
        return tokenizer