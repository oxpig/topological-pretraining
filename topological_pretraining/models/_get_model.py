from . import __dict__ as nn_dict


def get_model(name):
    return nn_dict[name]
