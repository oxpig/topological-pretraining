from _src.data.mol import MorganGenerator, Standardizer

import argparse
from pathlib import Path
import yaml
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

parser = argparse.ArgumentParser(description='Topological Pretraining')
parser.add_argument('--config', '-C', type=str, required=True, help='Path to the config file')
parser.add_argument('--data', '-D', type=str, required=True, help='Path to the data')
parser.add_argument('--output', '-o', type=str, default='output', help='Path to save')


def preprocess(args):
    pass

def pretrain(args):
    pass

def benchmark(args):
    pass

def evaluate(args):
    pass

def main():
    args = parser.parse_args()
    config = yaml.load(open(args.config), Loader=yaml.Loader)

    # Set ECFP Morgan generator
    morgan_settings = config['morgan'] if 'morgan' in config else {'radius': 2, 'fpSize': 2048, 'includeChirality': True}
    ecfp_generator = GetMorganGenerator(**morgan_settings)
    ecfp_generator = MorganGenerator(ecfp_generator)

    # Set standardizer
    standardizer = Standardizer()

    # Get ECFP fingerprints of datasets

    process = config['process']
    if process == 'preprocess':
        preprocess(args)
    elif process == 'pretrain':
        pretrain(args)
    elif process == 'benchmark':
        benchmark(args)
    elif process == 'evaluate':
        evaluate(args)
    elif process == 'unittest':
        # run unit tests
        pass
    else:
        raise ValueError(
            'Invalid process in config file. Must be one of preprocess,\
            pretrain, benchmark, evaluate, or unittest'
        )


if __name__ == "__main__":
    main()