from . import (
    BaseTokenizer,
    AtomGraphTokenizer, MorganGraphTokenizer,
    ECFP, FCFP, PDV, SNS
)
import torch

all_featurizers = {
    'AtomGraphTokenizer': AtomGraphTokenizer,
    'MorganGraphTokenizer': MorganGraphTokenizer,
    'ECFP': ECFP,
    'FCFP': FCFP,
    'PDV': PDV,
    'SNS': SNS
}


def read_from_dict(parameters: dict) -> BaseTokenizer:
    """
    Load featurizer from a dictionary.

    Parameters:
    ----------
    parameters : dict
        A dictionary containing the featurizer's parameters. The dictionary must include
        a 'name' key that specifies the type of featurizer to instantiate, along with any
        additional parameters required for that featurizer.

    Returns:
    -------
    BaseTokenizer
        An instance of the specified featurizer class, initialized with the provided parameters.

    Raises:
    ------
    ValueError
        If the 'name' key is not found in the parameters dictionary.
    """
    if 'name' not in parameters:
        raise ValueError('Tokenizer name not found in parameters.')
    featurizer = parameters.pop('name')
    featurizer = all_featurizers[featurizer](**parameters)
    featurizer.is_fitted_ = parameters['is_fitted_']
    return featurizer


def load_featurizer(path: str, parameters: bool = True) -> BaseTokenizer:
    """
    Load a featurizer from a file.

    Parameters:
    ----------
    path : str
        The file path to the saved featurizer. The file should be a PyTorch model file.
    parameters : bool, optional
        If True, the function will return a dictionary of parameters instead of the featurizer object.
        Defaults to True.

    Returns:
    -------
    BaseTokenizer or dict
        If `parameters` is True, loads a saved dictionary and initializes a featurizer from it.
        If `parameters` is False, returns the featurizer object directly.
    """
    featurizer = torch.load(path, weights_only=parameters)
    if parameters:
        return read_from_dict(featurizer)
    else:
        return featurizer