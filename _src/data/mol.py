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


class MorganGenerator:
    """
    Python wrapper for Morgan fingerprint generator.
    """
    def __init__(
        self,
        generator: GetMorganGenerator = GetMorganGenerator(
            radius=2,
            includeChirality=True,
            useBondTypes=True,
            includeRingMembership=True,
            fpSize=2048,
        ),
    ):
        self.generator = GetMorganGenerator(
            radius=2,
            includeChirality=True,
            useBondTypes=True,
            includeRingMembership=True,
        )

    def dense(
        self, mol: Chem.Mol|str, array: bool = False
    ) -> DataStructs.ExplicitBitVect|np.ndarray:
        """
        Generate a dense Morgan fingerprint for a molecule.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            The molecule.
        array : bool, optional
            Whether to return as an array. Defaults to False.

        Returns
        -------
        rdkit.DataStructs.cDataStructs.ExplicitBitVect
            The dense Morgan fingerprint.
        """
        if array:
            return self.generator.GetFingerprintAsNumPy(mol)
        else:
            return self.generator.GetFingerprint(mol)
              
    def sparse(self, mol: Chem.Mol) -> DataStructs.ExplicitBitVect:
        """
        Generate a sparse Morgan fingerprint for a molecule.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            The molecule.

        Returns
        -------
        rdkit.DataStructs.ExplicitBitVect
            The sparse Morgan fingerprint.
        """
        return self.generator.GetSparseFingerprint(mol)
    
    def bitinfo(self, mol: Chem.Mol) -> dict:
        """
        Get hashed identifiers mapped to atom indices and radii.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            The molecule.

        Returns
        -------
        dict
            Identifier map of molecule. Keys are hashed identifiers.
            Values are tuples of (atom index, radius).
        """
        ao = AdditionalOutput()
        ao.AllocateBitInfoMap()
        self.generator.GetSparseFingerprint(mol, additionalOutput=ao)
        return ao.GetBitInfoMap()
    
    def env_map(self, mol: Chem.Mol) -> dict:
        """
        Get array of hashed substructure identifiers mapped to atom indices and radii.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
        
        Returns
        -------
        np.ndarray
            Array of sparse hashed identifiers mapped to atom indices and radii.
            Rows are atom indices and columns are radii.

        """
        out = {}
        bitmap = self.bitinfo(mol)
        num_radii = range(self.radius + 1)
        num_atoms = range(mol.GetNumAtoms())
        out = np.zeros((num_atoms, num_radii))
        for bit, info in bitmap.items():
            for (atom, r) in info:
                out[atom, r] = bit
        
        inci = get_incidence(mol, radius=self.radius)
        missing_envs = np.where(out == 0)

        if len(missing_envs) > 0:
            for atom, radius in missing_envs:
                env = inci[radius, atom, :]
                env_matches = env[np.where((np.all(inci == env, axis=-1)))]
                env_matches = env_matches[np.where(env_matches != 0)]
                out[atom, radius] = env_matches[0] if len(env_matches) > 0 else 0

        return out

    @property
    def radius(self):
        return self.generator.GetOptions().radius
    
    @radius.setter
    def radius(self, value):
        self.generator.GetOptions().radius = value

    @property
    def fpsize(self):
        return self.generator.GetOptions().fpSize
    
    @fpsize.setter
    def fpsize(self, value):
        self.generator.GetOptions().fpSize = value

    @property
    def chirality(self):
        return self.generator.GetOptions().includeChirality
    
    @chirality.setter
    def chirality(self, value):
        self.generator.GetOptions().includeChirality = value

    @property
    def redundant_envs(self):
        return self.generator.GetOptions().includeRedundantEnvironments
    
    @redundant_envs.setter
    def redundant_envs(self, value):
        self.generator.GetOptions().includeRedundantEnvironments = value

    @property
    def counts(self):
        return self.generator.GetOptions().countSimulation
    
    @counts.setter
    def counts(self, value):
        self.generator.GetOptions().countSimulation = value

    @property
    def non_zero_invariants(self):
        return self.generator.GetOptions().nonZeroInvariants
    
    @non_zero_invariants.setter
    def non_zero_invariants(self, value):
        self.generator.GetOptions().nonZeroInvariants = value

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
        radius = self.generator.GetOptions().radius
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
        

    def __call__(self, mol: Chem.Mol) -> Chem.Mol:
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
    """
    Functions that operate on Morgan fingerprints.
    """

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
    
def get_incidence(mol: Chem.Mol, radius: int = 2) -> np.ndarray:
    """
    Get the incidence array of a molecule.
    Denotes atoms within cicular subgraphs of a given radius around each node.

    Parameters:
    ----------
        mol (Chem.Mol): RDKit molecule.
        radius (int): Radius of the incidence array.
    
    Returns:
    -------
        np.ndarray: Incidence array. Dims = radius x num_atoms x num_atoms.
        At each radius, the indexes of circular substructures around each atom are stored.
        E.g.,
            radius = 0: Identity matrix.
            radius = 1: Adjacency matrix.
            radius = 2: Adjacency matrix + 2nd degree neighbors.
            radius = 3: Adjacency matrix + 2nd and 3rd degree neighbors.
    """
    adjacency = Chem.GetAdjacencyMatrix(mol)
    inc = np.eye(adjacency.shape[0], dtype=int)
    if radius == 0:
        return inc

    inc = np.stack([inc, adjacency + inc], axis=0)
    if radius == 1:
        return inc

    to_add = inc[1]
    r = 2

    while True:
        pow_adj = np.linalg.matrix_power(adjacency, r)
        to_add = pow_adj + to_add
        inc = np.concatenate([inc, np.expand_dims(to_add, axis=0)], axis=0)
        if r == radius:
            break
        elif radius == -1 and np.all(to_add > 0):
            break
        else:
            r += 1
            continue

    inc = np.where(inc > 0, 1, 0)
    return inc
