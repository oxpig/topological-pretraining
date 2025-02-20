from . import PDV, SNS, ECFP, FCFP
from ..nn import pred_head
from .. import tokenizers 

from rdkit import Chem
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
import torch_geometric as pyg
import torch





possible_targets_ = {
    'SNS': {
        'head_type': 'binary',
        'target_class': SNS,
        'refit': False,
        'level': 'global',
        'input_type': 'molecule',
        'split_dependent': True,
        'run_dependent': False,
    },
    'ECFP': {
        'head_type': 'binary',
        'target_class': ECFP,
        'processing': [],
        'refit': False,
        'level': 'global',
        'input_type': 'molecule',
        'split_dependent': False,
        'run_dependent': False,
    },
    'FCFP': {
        'head_type': 'binary',
        'target_class': FCFP,
        'level': 'global',
        'input_type': 'molecule',
        'split_dependent': False,
        'run_dependent': False,
    },
    'PDV': {
        'head_type': 'regression',
        'target_class': PDV,
        'level': 'global',
        'input_type': 'molecule',
        'split_dependent': False,
        'run_dependent': False,
    },
}

class Targets(dict):

    def __init__(
        self, targets: dict[str, dict[str, str]] = {}, targets_path: str = None
    ):
        super(Targets, self).__init__(**targets)
        self.targets_path = targets_path
        if self.targets_path is not None:
            self.load()
        else:
            for target_name, target_kwargs in self.items():
                if target_name not in possible_targets_:
                    raise ValueError(f'Target {target_name} not supported.')
                self[target_name]['pipeline'] = possible_targets_[target_name]['target_class'](**target_kwargs)
                self[target_name]['prediction_head'] = possible_targets_[target_name]['head_type']
                self[target_name]['level'] = possible_targets_[target_name]['level']
                self[target_name]['input_type'] = possible_targets_[target_name]['input_type']

    def fit(self, data: tuple[list[Chem.Mol], list[pyg.data.Data]]):
        for target_name in self:
            input_type = self[target_name]['input_type']
            
            if input_type == 'molecule':
                x = torch.tensor(self[target_name]['pipeline'].fit_transform(data[0]))
            elif input_type == 'graph':
                x = torch.tensor(self[target_name]['pipeline'].fit_transform(data[1]))
            print(x.size())
            if self[target_name]['prediction_head'] == 'binary':
                zero_class = torch.zeros_like(x)
                zero_class[x == 0] = 1
                weights = torch.stack([
                    x.size(0) / (2 * zero_class.sum(dim=0)),
                    x.size(0) / (2 * x.sum(dim=0))
                ])
                weights.type(torch.float64)
                print(zero_class.sum(dim=0))
                print(x.sum(dim=0))
                print(weights)
                exit()

            elif self[target_name]['prediction_head'] == 'multiclass':
                weights = torch.tensor(x.size(0) / (x.size(1) * x.sum(dim=0)))
                weights.type(torch.float64)

            else:
                weights = None
            self[target_name]['class_weights'] = weights

        return self

    def transform(self, mol: Chem.Mol, graph: pyg.data.Data):

        for target_name in self:
            if target_name in graph:
                continue
            input_type = self[target_name]['input_type']
            level = self[target_name]['level']
            prediction_head = self[target_name]['prediction_head']
            if prediction_head == 'regression':
                dtype = torch.float32
            else:
                dtype = torch.long

            if input_type == 'molecule':
                X = mol
            else:
                X = graph
            y = self[target_name]['pipeline'].transform(X)

            if level == 'global':
                graph[target_name] = torch.tensor(y, dtype=dtype)
            elif level == 'node':
                y, mask = y
                graph[target_name] = torch.tensor(y, dtype=dtype)
                graph[f'{target_name}_mask'] = mask
            else:
                raise ValueError(f'Level {level} not supported.')

        return graph
    
    def to_dict(self):
        out = {}
        for target in self:
            out[target] = self[target].copy()
            out[target]['pipeline'] = self[target]['pipeline'].to_dict()
        return out 
    
    def save(self, targets_path: str = None):
        if targets_path is None:
            targets_path = self.targets_path
        else:
            self.targets_path = targets_path
        torch.save(self.to_dict(), targets_path)

    def load(self):
        targets = torch.load(self.targets_path, weights_only=True)
        for target in targets:
            self[target] = targets[target]
            self[target]['pipeline'] = tokenizers.read_from_dict(self[target]['pipeline'])
            self[target]['prediction_head'] = possible_targets_[target]['head_type']
            self[target]['level'] = possible_targets_[target]['level']
            self[target]['input_type'] = possible_targets_[target]['input_type']

    @property
    def is_fitted_(self):
        return all([
            self[target_name]['pipeline'].__sklearn_is_fitted__()
            for target_name in self
        ])

    