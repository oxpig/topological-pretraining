from _src.data.mol import MorganGenerator, Standardizer

import argparse
from pathlib import Path
import yaml
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from _src import preprocess

parser = argparse.ArgumentParser(description='Topological Pretraining')
parser.add_argument('--config', '-C', type=str, required=True, help='Path to the config file')
parser.add_argument('--data', '-D', type=str, required=True, help='Path to the data')
parser.add_argument('--output', '-o', type=str, default='output', help='Path to save')


def benchmark(args):
    # Load the data

    # Load the model / featurizer

    # Featurize the data

    # Hyperparameter tuning

    # Loop over splits
    ## PCA / dimensionality reduction
    ## Train the model
    ## Test the model
    pass

def evaluate(args):
    pass

def main():
    args = parser.parse_args()
    config: dict = yaml.load(open(args.config), Loader=yaml.Loader)
    process = config.pop('process')
    config['data'] = Path(args.data)
    config['output'] = Path(args.output)
    if 'verbose' not in config:
        config['verbose'] = False
    if process == 'preprocess':
        preprocess.preprocess(config=config)
    elif process == 'pretrain':
        pass
    elif process == 'benchmark':
        pass
    elif process == 'evaluate':
        pass
    else:
        raise ValueError('Invalid process')


if __name__ == "__main__":
    main()