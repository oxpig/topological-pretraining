from . import __dict__ as featurizers


def get_featurizer(name):
	return featurizers[name]
