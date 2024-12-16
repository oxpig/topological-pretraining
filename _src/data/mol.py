import numpy as np
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import (
    AdditionalOutput, FingeprintGenerator64, GetMorganGenerator
)
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
from tqdm import tqdm
import warnings

class SortAndSlice:
    """
    Class to sort and slice substructure identifiers from molecules.
    See:
        Dablander, M., Hanser, T., Lambiotte, R., Morris, G.M., 2024.
        Sort & Slice: a simple and superior alternative to hash-based folding for
        extended-connectivity fingerprints. Journal of Cheminformatics 16, 135.
        https://doi.org/10.1186/s13321-024-00932-y

    Parameters:
    ----------
        molecules (list[Chem.Mol]): List of RDKit molecules.
        generator (FingeprintGenerator64): RDKit fingerprint generator.
        fpsize (int): Length of the output vector.
        verbose (bool): Whether to print progress.

    Attributes:
    ----------
        generator (FingeprintGenerator64): RDKit fingerprint generator.
        ao (AdditionalOutput): RDKit fingerprint additional output.
        verbose (bool): Whether to print progress.
        identifiers (dict[str, int]): Dictionary of identifiers and their counts.
        encoder (dict[str, int]): Dictionary of identifiers and their enumerated values.

    Example:
    -------
    >>> from rdkit import Chem
    >>> from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
    >>> smiles = ['CCO', 'CCN', 'CCC']
    >>> molecules = [Chem.MolFromSmiles(s) for s in smiles]
    >>> generator = GetMorganGenerator()
    >>> sas = SortAndSlice(molecules, generator, fpsize=128, verbose=True)
    >>> encoded_moleculess = sas(molecules)
    """
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

class Standardizer:
    """
    Class to standardize rdkit molecules.

    Parameters:
    ----------
        sanitize (bool): Whether to sanitize the molecule.
        cleanup (bool): Whether to cleanup the molecule.
        fragment_parent (bool): Whether to fragment the molecule.
        neutralize (bool): Whether to neutralize the molecule.
        reionize (bool): Whether to reionize the molecule after neutralization.
        canonical_tautomer (bool): Whether to canonicalize the tautomer.

    Attributes:
    ----------
        sanitize (bool): Whether to sanitize the molecule.
        fragment_parent (bool): Whether to fragment the molecule.
        neutralize (bool): Whether to neutralize the molecule.
        reionize (bool): Whether to reionize the molecule after neutralization.
        canonical_tautomer (bool): Whether to canonicalize the tautomer.

    Example:
    -------
        >>> from rdkit import Chem
        >>> mol = Chem.MolFromSmiles('c1ccccc1')
        >>> standardizer = Standardizer()
        >>> mol = standardizer(mol)
    """
    def __init__(
        self,
        sanitize: bool = True,
        fragment_parent: bool = True,
        neutralize: bool = True,
        reionize: bool = False,
        canonical_tautomer: bool = True,
    ):
        self.sanitize = sanitize
        self.fragment_parent = fragment_parent
        self.neutralize = neutralize
        self.reionize = reionize
        self.canonical_tautomer = canonical_tautomer

    def standardize(self, mol: Chem.Mol) -> Chem.Mol:
        """
        Standardizes a molecule.

        Parameters:
        ----------
            mol (Chem.Mol|str): RDKit molecule or a SMILES string.

        Returns:
        -------
            Chem.Mol: Standardized RDKit molecule.
        """
        mol.UpdatePropertyCache()
        if self.sanitize:
            mol = self.run_sanitize(mol)

        if self.fragment_parent:
            mol = self.run_fragment_parent(mol)
        
        if self.neutralize:
            mol = self.run_neutralize(mol)

        if self.reionize:
            self.run_reionize(mol)
        
        if self.canonical_tautomer:
            mol = self.run_canonical_tautomer(mol)
        
        mol.UpdatePropertyCache()
        return mol
        

    def __call__(self, mol: Chem.Mol|str) -> Chem.Mol:
        """
        Standardizes a molecule.

        Parameters:
        ----------
            mol (Chem.Mol): RDKit molecule.

        Returns:
        -------
            Chem.Mol: Standardized RDKit molecule.
        """
        return self.standardize(mol)

    def run_sanitize(self, mol: Chem.Mol) -> Chem.Mol:
        """
        Sanitizes a molecule.
        See: https://sourceforge.net/p/rdkit/mailman/message/31897681/

        Parameters:
        ----------
            mol (Chem.Mol): RDKit molecule.

        Returns:
        -------
            Chem.Mol: Sanitized RDKit molecule.
        """
        return Chem.SanitizeMol(mol)
    
    
    def run_fragment_parent(self, mol: Chem.Mol) -> Chem.Mol:
        """
        For molecules with multiple fragments, chooses the largest fragment.

        Parameters:
        ----------
            mol (Chem.Mol): RDKit molecule.

        Returns:
        -------
            Chem.Mol: Fragmented RDKit molecule.
        """
        return rdMolStandardize.FragmentParent(mol)

    def run_neutralize(self, mol: Chem.Mol) -> Chem.Mol:
        """
        Neutralizes a molecule.

        Parameters:
        ----------
            mol (Chem.Mol): RDKit molecule.

        Returns:
        -------
            Chem.Mol: Neutralized RDKit molecule.
        """
        uncharger = rdMolStandardize.Uncharger()
        return uncharger.uncharge(mol)
    
    def run_canonical_tautomer(self, mol: Chem.Mol) -> Chem.Mol:
        """
        Canonicalizes a tautomer.

        Parameters:
        ----------
            mol (Chem.Mol): RDKit molecule.

        Returns:
        -------
            Chem.Mol: Canonicalized RDKit molecule.
        """
        return rdMolStandardize.CanonicalTautomer(mol)

    def run_reionize(self, mol: Chem.Mol) -> Chem.Mol:
        """
        Reionizes a molecule.

        Parameters:
        ----------
            mol (Chem.Mol): RDKit molecule.

        Returns:
        -------
            Chem.Mol: Reionized RDKit molecule.
        """
        return rdMolStandardize.Reionize(mol)

class FPOperations:

    def tanimoto(fp1: DataStructs.ExplicitBitVect, fp2: DataStructs.ExplicitBitVect) -> float:
        """
        Calculates the Tanimoto similarity between two fingerprints.

        Parameters:
        ----------
            fp1 (Data): Fingerprint 1.
            fp2 (np.ndarray): Fingerprint 2.

        Returns:
        -------
            float: Tanimoto similarity.
        """
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    
    def bulk_tanimoto(
        fp: DataStructs.ExplicitBitVect,
        fp_list: list[DataStructs.ExplicitBitVect]
    ):
        """
        Calculates the Tanimoto similarity between a fingerprint and a list of
        fingerprints.

        Parameters:
        ----------
            fp (DataStructs.ExplicitBitVect): Fingerprint.
            fp_list (list[DataStructs.ExplicitBitVect]): List of fingerprints.

        Returns:
        -------
            list[float]: Tanimoto similarities between the fingerprint and the list of
            fingerprints.
        """
        return DataStructs.BulkTanimotoSimilarity(fp, fp_list)
    
    def list_tanimoto(
        fps1: list[DataStructs.ExplicitBitVect],
        fps2: list[DataStructs.ExplicitBitVect],
    ):
        """
        Calculates the Tanimoto similarity between two lists of fingerprints.

        Parameters:
        ----------
            fps1 (list[DataStructs.ExplicitBitVect]): Set of fingerprints 1.
            fps2 (list[DataStructs.ExplicitBitVect]): Set of fingerprints 2.

        Returns:
        -------
            np.ndarray: Tanimoto similarities. Dims = len(fps1) x len(fps2).
        """
        similarities = np.zeros((len(fps1), len(fps2)))
        for i, fp1 in enumerate(fps1):
            similarities[i] = DataStructs.BulkTanimotoSimilarity(fp1, fps2)
        return similarities
    
    def pairwise_tanimoto(fps: list[DataStructs.ExplicitBitVect]):
        """
        Calculates the pairwise Tanimoto similarity within a list of fingerprints.

        Parameters:
        ----------
            fps (list[DataStructs.ExplicitBitVect],): List of fingerprints.
        
        Returns:
        -------
            np.ndarray: Pairwise Tanimoto similarities. Dims = len(fps) x len(fps).
        """
        similarities = np.zeros((len(fps), len(fps)))
        for i in range(len(fps)):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
            similarities[i, :i], similarities[:i, i] = sims, sims
        return similarities
    
    def butina(fps: list[DataStructs.ExplicitBitVect], threshold: float = 0.65):
        """
        Performs the Butina clustering algorithm.

        Parameters:
        ----------
            fps (list[DataStructs.ExplicitBitVect]): List of fingerprints.
            threshold (float): Tanimoto similarity threshold.

        Returns:
        -------
            np.ndarray: Cluster assignments for each fingerprint.
        """
        assert threshold >= 0 and threshold <= 1, 'Threshold must be between 0 and 1.'
        distances = 1 - FPOperations.pairwise_tanimoto(fps)
        clusters = Butina.ClusterData(distances, distThresh=threshold, isDistData=True, nPts=len(fps))
        
        molecule_clusters = np.zeros(len(fps))
        for i, cluster in enumerate(clusters):
            for j in cluster:
                molecule_clusters[j] = i

        return molecule_clusters