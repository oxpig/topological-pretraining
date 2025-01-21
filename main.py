import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from _src.data.mol import MorganGenerator, Standardizer

import argparse
from pathlib import Path
import yaml
from _src import benchmark, preprocess, evaluate

from dotenv import load_dotenv

load_dotenv()

parser = argparse.ArgumentParser(description='Topological Pretraining')
parser.add_argument('--config', '-C', type=str, required=True, help='Path to the config file')
parser.add_argument('--data', '-D', type=str, required=True, help='Path to the data')
parser.add_argument('--output', '-o', type=str, default='output', help='Path to save')

def load_base_config() -> dict:
    config_dir = Path(__file__).parent / 'config' / 'base'
    base_config = {}
    for file in config_dir.iterdir():
        if file.suffix == '.yaml':
            with open(file, 'r') as f:
                config = yaml.load(f, Loader=yaml.Loader)
                base_config.update(config)
    return base_config


def main():
    args = parser.parse_args()
    config = load_base_config()
    config.update(yaml.load(open(args.config), Loader=yaml.Loader))
    process = config.pop('process')
    config['data'] = Path(args.data)
    config['results'] = Path(args.output)
    if 'verbose' not in config:
        config['verbose'] = False
    if process == 'preprocess':
        preprocess.preprocess(config=config)
    elif process == 'pretrain':
        pass
    elif process == 'benchmark':
        benchmark.benchmark(config=config)
    elif process == 'evaluate':
        evaluate.evaluate(config=config)
    else:
        raise ValueError('Invalid process')


if __name__ == "__main__":
    main()