from topological_pretraining.data.mol import MorganGenerator, SortAndSlice, MolDesc
from topological_pretraining.featurization.base import BaseFeaturizer

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganFeatureAtomInvGen

from typing import Optional, TYPE_CHECKING

morgan_feat_inv = GetMorganFeatureAtomInvGen()

default_descriptors = [i[0] for i in Descriptors._descList]

class PDV(BaseFeaturizer):
    """
    Featurizer that computes molecular descriptors using RDKit's MolDesc.
    This featurizer is designed to compute a set of predefined molecular descriptors
    for each molecule in the input data.

    Parameters:
    ----------
    transform_kwargs : dict
        Keyword arguments for the MolDesc transformation function.
        This can include parameters such as `descriptors` to specify which descriptors to compute.
    verbose : bool, optional
    """
    precomputed = True 
    def _transform_base(self, **kwargs):
        if 'descriptors' not in kwargs:
            kwargs['descriptors'] = default_descriptors 
        return MolDesc(verbose=self.verbose, **kwargs)

class ECFP(BaseFeaturizer):
    """
    Featurizer that computes Extended Connectivity Fingerprints (ECFP) using RDKit's Morgan fingerprints.
    This featurizer is designed to compute ECFP fingerprints for each molecule in the input data.

    Parameters:
    ----------
    transform_kwargs : dict
        Keyword arguments for the MorganGenerator transformation function.
        This can include parameters such as `fpsize` to specify the size of the fingerprint.
        See `MorganGenerator` in `topological_pretraining/data/mol.py` for more details.
    verbose : bool, optional
        If True, the featurizer will print additional information during processing.
    """
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
    """
    Modified version of ECFP that uses a different fingerprint generator.
    This featurizer computes Functional Connectivity Fingerprints (FCFP) using RDKit's Morgan fingerprints.

    Parameters:
    ----------
    transform_kwargs : dict
        Keyword arguments for the MorganGenerator transformation function.
        This can include parameters such as `fpsize` to specify the size of the fingerprint.
        See `MorganGenerator` in `topological_pretraining/data/mol.py` for more details.
    verbose : bool, optional
        If True, the featurizer will print additional information during processing.
    """
    precomputed = True
    is_fitted_ = False
    fixed_transform_kwargs = {'atom_inv': morgan_feat_inv}

class SNS(BaseFeaturizer):
    """
    Molecular featurizer that computes a sorted and sliced version of Morgan fingerprints.
    This featurizer is designed to compute Morgan fingerprints, sort them, and slice them 
    to a specified size.

    Parameters:
    ----------
    identifiers : dict, optional
        A pre-definced SortAndSlice dictionary of identifiers for the molecules.
    encoder : dict, optional
        A dictionary specifying a pre-determined encoding scheme for the fingerprints. 
        This can include parameters such as `fpsize` to specify the size of the fingerprint.
    transform_kwargs : dict, optional
        Keyword arguments for the SortAndSlice transformation function.
        This can include parameters such as the `morgan_kwargs` parameter used to pass additional 
        parameters to the MorganGenerator and `fpsize` to specify the size of the fingerprint.
        See `SortAndSlice` in `topological_pretraining/data/mol.py` for more details.
    verbose : bool, optional
    """
    is_fitted_ = False
    precomputed = False

    def __init__(self, identifiers: dict = {}, encoder: dict = {}, **kwargs):
        super().__init__(**kwargs)
        self.transform.identifiers = identifiers
        self.transform.encoder = encoder

    def _transform_base(self, **kwargs):
        """
        Create a SortAndSlice transformation that uses a MorganGenerator.

        Parameters:
        ----------
        morgan_kwargs : dict, optional
            Keyword arguments for the MorganGenerator. 
        **kwargs : dict, optional
            Additional keyword arguments for the SortAndSlice transformation.
            This can include parameters such as `fpsize` to specify the size of the fingerprint.

        Returns:
        -------
        SortAndSlice
            An instance of the SortAndSlice transformation that uses a MorganGenerator.
        """
        if 'morgan_kwargs' in kwargs:
            morgan: dict = kwargs.pop('morgan_kwargs')
        else:
            morgan: dict = {}
        morgan = MorganGenerator(**morgan)
        return SortAndSlice(generator=morgan, **kwargs)
        
    def fit(self, mols: list[Chem.Mol], y: Optional[np.ndarray] = None) -> None:
        """
        Fit the featurizer to the provided molecules and their corresponding labels.

        Parameters:
        ----------
        mols : list[Chem.Mol]
            A list of RDKit molecule objects to be tokenized.
        y : np.ndarray, optional
            An optional array of labels corresponding to the molecules.
        
        Returns:
        -------
        self : SNS
            Returns the fitted featurizer instance.
        """
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
        """
        Convert the featurizer's parameters and state to a dictionary.

        Returns:
        -------
        dict
            A dictionary containing the featurizer's name, fitted status, and 
            transformation parameters. This includes the identifiers and 
            encoder used in the SortAndSlice transformation.
        """
        params = super().to_dict()
        params['identifiers'] = self.identifiers
        params['encoder'] = self.encoder
        return params
    
    def preprocess(self, mols: list[Chem.Mol]) -> list[np.ndarray]:
        """
        Preprocess the input molecules by calculating the atomic environment identifiers 
        for each molecule. Useful for preparing molecules for repeated sort and slice operations
        under different conditions, e.g., embedding with SNS fitted on with new data.
        """
        return [self.transform.generator.environments(mol) if mol is not None else mol for mol in mols]

    

    
