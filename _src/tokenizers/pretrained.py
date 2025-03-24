from _src.tokenizers.base import BaseTokenizer
from _src.tokenizers.load import read_from_dict
from _src.nn.pred_head import PredHead
from _src.nn import get_nn

from rdkit import Chem
import torch
import torch_geometric as pyg
from typing import Literal

class PreTrainedModel(torch.nn.Module):
    
    def __init__(
        self,
        path: str|None = None,
        params: dict|None = None,
        device: str = None,
        asarray: bool = True
    ):
        super(PreTrainedModel, self).__init__()
        self.asarray = asarray
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        self.device = device
        if params is not None:
            self.from_dict(params)
        elif path is not None:
            self.load(path=path)
        else:
            raise ValueError('Either path or params must be provided.')
        
    def from_dict(self, params: dict):
        tokenizer = params.pop('tokenizer')
        self.tokenizer = read_from_dict(tokenizer)
        main_model = params.pop('main')
        main_cls = main_model['cls']
        main_cls = get_nn(main_cls)
        self.model = main_cls(**main_model['kwargs'])
        self.model.load_state_dict(main_model['state'])
        self.model.eval()
        self.heads = torch.nn.ModuleDict()
        self.heads_kwargs = {}
        for head in params.get('heads', {}):
            head_cls = params['heads'][head]['cls']
            head_cls = get_nn(head_cls)
            head_kwargs = params['heads'][head]['kwargs']
            head_state = params['heads'][head]['state']
            self.heads[head] = head_cls(**head_kwargs)
            self.heads[head].load_state_dict(head_state)
            self.heads[head].eval()
            self.heads_kwargs[head] = head_kwargs

        self.to_device()

    def embed(self, x: Chem.Mol):
        return self.model(x)

    def forward(self, x: Chem.Mol):
        x = self.embed(x)
        if self.asarray:
            x = x.detach().cpu().numpy()
        return x
    
    def get_head_preds(
        self, x: Chem.Mol|list[Chem.Mol]
    ):
        x = self.tokenize(x)
        x = self.model(x)
        preds = {}
        for target in self.heads:
            preds[target] = self.heads[target](x)
        return preds
    
    @property
    def model_cls(self):
        return self.model.__class__.__name__
    
    @property
    def model_state_dict(self):
        return self.model.state_dict()
    
    def to_dict(self):
        params = {
            'tokenizer': self.tokenizer.to_dict(),
            'main': self.model.state_dict(),
            'main_class': self.model_cls,
            'main_kwargs': self.model_kwargs,
        }
        params['heads'] = {}
        for target in self.heads:
            params['heads'][target] = {
                'state': self.heads[target].state_dict(),
                'cls': self.heads[target].__class__.__name__,
                'kwargs': self.heads_kwargs[target],
            }
        return params

    def save(self, path):
        params = self.to_dict()
        torch.save(params, path)
        self.path = path

    def to_device(self, device = None):
        if device is None:
            device = self.device
        else:
            self.device = device
        super().to(device)
        
    def load(self, path: str):
        self.path = path
        params = torch.load(path, weights_only=True, map_location='cpu')
        self.from_dict(params)

    def tokenize(self, x: Chem.Mol|list[Chem.Mol]):
        return self.tokenizer.transform(x)

class PreTrainedGNN(PreTrainedModel):

    def __init__(
        self,
        path: str|None = None,
        params: dict|None = None,
        embed_state: Literal['node', 'global', 'all'] = 'global',
        layer_pool_type: slice|int|Literal['last', 'sum', 'mean', 'max', 'concat'] = None,
        device: str = None,
        asarray: bool = True,
        **kwargs,
    ):
        self.layer_pool_type = layer_pool_type
        self.embed_state = embed_state
        super(PreTrainedGNN, self).__init__(path=path, params=params, device=device, asarray=asarray)
        
    def from_dict(self, params: dict):
        super().from_dict(params)
        if self.layer_pool_type is not None:
            self.model.layer_pool_type = self.layer_pool_type
    
    def embed(self, mol: Chem.Mol|list[Chem.Mol], embed_state: Literal['node', 'global', 'all'] = None):
        if embed_state is not None:
            self.embed_state = embed_state
        graph = self.tokenize(mol)
        if isinstance(graph, list):
            graph = pyg.data.Batch.from_data_list(graph)
        graph = graph.to(self.device)
        x = self.model(
            x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr,
            batch=graph.batch, global_idx=graph.get('global_idx')
        )
        if self.embed_state == 'node':
            return x['final_state']
        elif self.embed_state == 'global':
            return x['global_state']
        elif self.embed_state == 'all':
            return x
        else:
            raise ValueError(
                f'Invalid embed_state {self.embed_state}. '\
                f'Must be one of "node", "global", or "all".'
            )


class PreTrainedTokenizer(BaseTokenizer):

    is_fitted_ = True
    precomputed = True

    def _transform_base(self, **kwargs):
        if kwargs.get('gnn', False):
            return PreTrainedGNN(**kwargs)
        else:
            return PreTrainedModel(**kwargs)
    
    def to_dict(self):
        params = super().to_dict()
        params.update(self.transform.to_dict())
        return params
    
    def set_embed_state(
        self, embed_state: Literal['node', 'global', 'all']
    ):
        self.transform.embed_state = embed_state

