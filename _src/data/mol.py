import numpy as np
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import (
    AdditionalOutput, FingeprintGenerator64, GetMorganGenerator
)
from rdkit.Chem.MolStandardize import rdMolStandardize
from tqdm import tqdm
import warnings

class SortAndSlice:

    def __init__(
        self,
        molecules: list[Chem.Mol],
        generator: FingeprintGenerator64,
        fpsize: int = 2048,
        verbose: bool = False,
    ):
        self.generator = generator
        self.ao = AdditionalOutput()
        self.ao.AllocateBitInfoMap()
        self.verbose = verbose

        self.get_identifiers(molecules)
        self.set_encoder(fpsize)

    def get_identifiers(self, molecules: list[Chem.Mol]) -> dict[str, int]:
        """
        Gets and sorts identifiers from molecule data. Sets the identifiers attribute.
        
        Parameters:
        ----------
            molecules (list[Chem.Mol]): List of RDKit molecules.

        Sets:
        ------
            identifiers (dict[str, int]): Dictionary of identifiers and their counts.
        """
        identifiers = {}
        pbar = tqdm(total=len(molecules), desc='Collecting identifiers', disable=not self.verbose)
        for mol in molecules:
            self.generator.GetSparseFingerprint(mol, additionalOutput=self.ao)
            bitmap = self.ao.GetBitInfoMap()
            for identifier in bitmap:
                count = identifiers.get(identifier, 0)
                identifiers[identifier] = count + 1
            pbar.update(1)
        pbar.close()
        self.identifiers = dict(sorted(identifiers.items(), key=lambda x: x[1], reverse=True))

    def set_encoder(self, fpsize: int):
        """
        Slices substructure identifiers to a specific length enumerates them.
        Sets the encoder attribute to the enumerated identifiers.

        Parameters:
        ----------
            fpsize (int): Length of the output vector.

        Sets:
        ------
            encoder (dict[str, int]): Dictionary of identifiers and their enumerated values.
        """
        if self.verbose:
            print(f'Setting bit length of encoder to a max of {fpsize}.')
        encoder = {}
        for i, k in enumerate(self.identifiers.keys()):
            if i >= fpsize:
                break
            encoder[k] = i

        self.encoder = encoder
        if len(encoder) < fpsize:
            warnings.warn(
                f'Fewer observed substructures than fpsize.\
                Encoder is only {len(encoder)} bits long.'
            )

        if self.verbose:
            print(f'Encoder set to {len(encoder)} bits.')

    def encode(self, mol: Chem.Mol) -> np.ndarray:
        """
        Encodes a molecule into a binary sort and slice vector.

        Parameters:
        ----------
            mol (Chem.Mol): RDKit molecule.

        Returns:
        -------
            np.ndarray: Binary vector indicating substructure presence.
        """
        self.generator.GetSparseFingerprint(mol, additionalOutput=self.ao)
        bitmap = self.ao.GetBitInfoMap()
        out = np.zeros(len(self.encoder))
        for identifier in bitmap:
            if identifier in self.encoder:
                out[self.encoder[identifier]] = 1
        return out

    def __call__(self, molecules: list[Chem.Mol]|Chem.Mol) -> np.ndarray:
        """
        Encodes a list of molecules into a binary sort and slice matrix.

        Parameters:
        ----------
            molecules (list[Chem.Mol]): List of RDKit molecules.

        Returns:
        -------
            np.ndarray: Binary matrix indicating substructure presence.
        """
        if isinstance(molecules, Chem.Mol):
            molecules = [molecules]
        pbar = tqdm(total=len(molecules), desc='Encoding molecules', disable=not self.verbose)
        out = np.zeros((len(molecules), len(self.encoder)))
        for i, mol in enumerate(molecules):
            out[i] = self.encode(mol)
            pbar.update(1)
        pbar.close()
        return out
    
    def __repr__(self):
        return f'SortAndSlice(fpsize={len(self.encoder)})'
    