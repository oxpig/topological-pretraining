import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
from pathlib import Path
import yaml
from _src import benchmark, preprocess, pretrain

from dotenv import load_dotenv

load_dotenv()

parser = argparse.ArgumentParser(description='Topological Pretraining')
parser.add_argument('--config', '-C', type=str, required=True, help='Path to the config file')
parser.add_argument('--base_config', '-B', type=str, default=None, help='Base config')
parser.add_argument('--data', '-D', type=str, required=True, help='Path to the data')
parser.add_argument('--output', '-o', type=str, default='output', help='Path to save')

def load_base_settings() -> dict:
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
    config = load_base_settings()
    if args.base_config:
        exp_base_config = yaml.load(open(args.base_config), Loader=yaml.Loader)
    else:
        exp_base_config = {}
    exp_config = yaml.load(open(args.config), Loader=yaml.Loader)
    config.update(exp_base_config)
    config.update(exp_config)
    config['path'] = args.config
    if 'model' in config:
        config.update(config[config['model']])
    process = config.pop('process')
    if 'name' not in config:
        config['name'] = Path(args.config).stem
    config['data'] = Path(args.data)
    config['results'] = Path(args.output)

    config['seed'] = config.get('seed', 42)

    if 'verbose' not in config:
        config['verbose'] = False
    if process == 'preprocess':
        preprocess.preprocess(config=config)
    elif process == 'pretrain':
        pretrain.pretrain(config=config)
    elif process == 'benchmark':
        benchmark.benchmark(config=config)
    elif process == 'pretrain_autoencoder':
        pretrain.pretrain_autoencoder(config=config)
    else:
        raise ValueError('Invalid process')


if __name__ == "__main__":
    main()