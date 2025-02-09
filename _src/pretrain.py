from .utils import get_model, get_tokenizer
from .data.utils import load_dataset
from .data.datasets import BaseDataset


def pretrain(config: dict):
    """
    Run the pretraining process.
    """
    name: str = config['name']
    data_path: str = config['data']
    results_path: str = config['results']
    verbose: bool = config['verbose']
    pretrain_data: list[str]|str = config['pretrain_data']
    tokenizer_class = get_tokenizer(config['tokenizer'])
    tokenizer_kwargs = config.get('transform_kwargs', {})

    model_class = get_model(config['model'])
    model_kwargs = config.get('model_kwargs', {})
    print(f'\n##################################################\n') if verbose else None
    print(f'Pretraining run {name}') if verbose else None
    print(f'Tokenizer: {config['tokenizer']}') if verbose else None
    print(f'Model: {config['model']}') if verbose else None
    print(f'Data: {pretrain_data}') if verbose else None

    # Load the dataset
    dataset: BaseDataset = load_dataset(data_path, pretrain_data, verbose=verbose)
