# script for running experiments with different configs
# used for preprocessing, pretraining, and benchmarking on cluster
import argparse
import warnings
from pathlib import Path

import yaml
from dotenv import load_dotenv

from topological_pretraining import benchmark, preprocess, pretrain

warnings.filterwarnings('ignore', category=FutureWarning)
load_dotenv()

parser = argparse.ArgumentParser(description='Topological Pretraining')
parser.add_argument(
	'--config', '-C', type=str, required=True, help='Path to the config file'
)
parser.add_argument('--base_config', '-B', type=str, default=None, help='Base config')
parser.add_argument('--data', '-D', type=str, required=True, help='Path to the data')
parser.add_argument('--output', '-o', type=str, default='output', help='Path to save')
parser.add_argument(
	'--model_path', '-m', type=str, default=None, help='Path to the pretrained model'
)


def load_base_settings() -> dict:
	config_dir = Path(__file__).parent / 'config' / 'base'
	base_config = {}
	for file in config_dir.iterdir():
		if file.suffix == '.yaml':
			with open(file) as f:
				config = yaml.safe_load(f)
				base_config.update(config)
	return base_config


def main():
	args = parser.parse_args()
	config = load_base_settings()
	exp_base_config = yaml.safe_load(args.base_config) if args.base_config else {}
	exp_config = yaml.safe_load(args.config)
	config.update(exp_base_config)
	config.update(exp_config)
	config['path'] = args.config
	if 'model' in config:
		config.update(config[config['model']])
	process = config.pop('process', None)
	if process is None:
		raise ValueError('Process not specified in config')
	if 'name' not in config:
		config['name'] = Path(args.config).stem
	config['data'] = Path(args.data)
	config['results'] = Path(args.output)
	config['model_path'] = Path(args.model_path) if args.model_path else None
	config['seed'] = config.get('seed', 42)

	if 'verbose' not in config:
		config['verbose'] = False
	if process == 'preprocess':
		preprocess.preprocess(config=config)
	elif process == 'pretrain':
		pretrain.pretrain(config=config)
	elif process == 'benchmark':
		print(config.get('featurizer'))
		if config.get('featurizer') == 'PreTrainedFeaturizer':
			if args.model_path is None:
				raise ValueError(
					'Pretrained model path required for benchmarking a pretrained model\nPlease specify the path to the pretrained model using the --model_path argument'
				)
			config['featurizer_kwargs']['path'] = args.model_path

		benchmark.benchmark(config=config)
	elif process == 'pretrain_autoencoder':
		pretrain.pretrain_autoencoder(config=config)
	else:
		raise ValueError('Invalid process')


if __name__ == '__main__':
	main()
