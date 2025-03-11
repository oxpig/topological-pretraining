from ..data.mol import MorganGenerator, SortAndSlice, MolDesc
from .base import BaseTokenizer, SelectAll


import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from rdkit.Chem.rdFingerprintGenerator import GetMorganFeatureAtomInvGen

from sklearn.feature_selection import SelectKBest

from typing import Callable, Literal, Optional

morgan_feat_inv = GetMorganFeatureAtomInvGen()

default_descriptors = [i[0] for i in Descriptors._descList]

class PDV(BaseTokenizer):
    precomputed = True
    def _transform_base(self, **kwargs):
        if 'descriptors' not in kwargs:
            kwargs['descriptors'] = default_descriptors 
        return MolDesc(verbose=self.verbose, **kwargs)

class ECFP(BaseTokenizer):

    is_fitted_ = False
    precomputed = True
    def _transform_base(self, **kwargs):
        gen = MorganGenerator(verbose=self.verbose, **kwargs)
        gen.asarray = True
        return gen

    @property
    def fpsize(self):
        return self.transform.fpsize
    
    @fpsize.setter
    def fpsize(self, value: int):
        self.transform.fpsize = value

class FCFP(ECFP):
    precomputed = True
    is_fitted_ = False
    fixed_transform_kwargs = {'atom_inv': morgan_feat_inv}

class SNS(BaseTokenizer):
    is_fitted_ = False
    precomputed = False

    def __init__(self, identifiers: dict = {}, encoder: dict = {}, **kwargs):
        super().__init__(**kwargs)
        self.transform.identifiers = identifiers
        self.transform.encoder = encoder

    def _transform_base(self, **kwargs):
        if 'morgan_kwargs' in kwargs:
            morgan: dict = kwargs.pop('morgan_kwargs')
        else:
            morgan: dict = {}
        morgan = MorganGenerator(**morgan)
        return SortAndSlice(generator=morgan, **kwargs)
        
    def fit(self, mols: list[Chem.Mol], y: Optional[np.ndarray] = None) -> None:
        self = super().fit(mols=mols, y=y)
        self.transform.clear()
        self.transform.update(mols)
        self.transform.sort()
        self.transform.slice()
        return self
        
    @property
    def fpsize(self):
        return self.transform.fpsize
    
    @property
    def encoder(self):
        return self.transform.encoder
    
    @property
    def identifiers(self):
        return self.transform.identifiers
    
    @fpsize.setter
    def fpsize(self, value: int):
        self.transform.slice(value)

    def to_dict(self):
        params = super().to_dict()
        params['identifiers'] = self.identifiers
        params['encoder'] = self.encoder
        return params
    
    def preprocess(self, mols: list[Chem.Mol]) -> list[np.ndarray]:
        return [self.transform.generator.environments(mol) for mol in mols]

    

    
