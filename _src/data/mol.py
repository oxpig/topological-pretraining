import numpy as np
from matplotlib import colors as mpl_colors
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.rdFingerprintGenerator import (
    AdditionalOutput, AtomInvariantsGenerator, BondInvariantsGenerator,
    GetMorganGenerator
)
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
from rdkit import RDLogger
from tqdm import tqdm
import warnings

"""
TODO: change print statements to logging
"""

RDLogger.DisableLog('rdApp.*')
default_atom_colors = {
    'C': '#BFBFBF', 'N': '#0000FF', 'O': '#FF0000',
    'F': '#00FF00', 'Cl': '#00FFFF', 'Br': '#FF00FF',
    'I': '#FFFF00', 'S': '#FFA500', 'P': '#800080',
}
for atom, color in default_atom_colors.items():
    default_atom_colors[atom] = mpl_colors.to_rgb(color)
default_atom_colors['H'] = (1,1,1)

class MorganGenerator:
    """
    Python wrapper for Morgan fingerprint generator.
    """
    def __init__(
        self,
        radius: int = 2,
        fpsize: int = 2048,
        chirality: bool = True,
        count_sim: bool = False,
        bond_types: bool = True,
        non_zero_inv: bool = False,
        rings: bool = True,
        count_bounds = None,
        atom_inv = None,
        bond_inv = None,
        redundant_envs: bool = False,
        asarray: bool = True,
        verbose: bool = False
    ):
        self.generator = GetMorganGenerator(
            radius=radius,
            countSimulation=count_sim,
            includeChirality=chirality,
            useBondTypes=bond_types,
            onlyNonzeroInvariants=non_zero_inv,
            includeRingMembership=rings,
            countBounds=count_bounds,
            fpSize=fpsize,
            atomInvariantsGenerator=atom_inv,
            bondInvariantsGenerator=bond_inv,
            includeRedundantEnvironments=redundant_envs
        )
        self.asarray = asarray
        self.verbose = verbose


    def __call__(
            self, mol: Chem.Mol|list[Chem.Mol]
        ) -> DataStructs.ExplicitBitVect|np.ndarray:
        if mol is None:
            return None
        if isinstance(mol, Chem.Mol):
            mol = [mol]
        verbosity = self.verbose if len(mol) > 1 else False
        out = []
        self.pbar = tqdm(
            total=len(mol), disable=not verbosity,
            desc='Generating fingerprints'
        )
        for idx, m in enumerate(mol):
            m = self.dense(m, array=self.asarray) if m is not None else np.full(self.fpsize, np.nan)
            out.append(m)
            self.pbar.update()
        self.pbar.close()
        self.pbar = None
        if self.asarray:
            out = np.array(out)
        return out

    def dense(
        self, mol: Chem.Mol, array: bool = False
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
    
    def environments(self, mol: Chem.Mol, ) -> dict:
        """
        Get array of hashed substructure identifiers mapped to atom indices and radii.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            The molecule.
        
        Returns
        -------
        np.ndarray
            Array of sparse hashed identifiers mapped to atom indices and radii.
            Rows are atom indices and columns are radii.

        """
        bitmap = self.bitinfo(mol)
        out = np.zeros((mol.GetNumAtoms(), self.radius + 1))
        for bit, info in bitmap.items():
            for (atom, r) in info:
                out[atom, r] = bit
        
        inci = get_incidence(mol, radius=self.radius)

        # Fill in missing environments for atoms with duplicate environments.
        # This needs to be performed regardless, as includeRedundantEnvironments
        # assigns duplicate environments with different central atoms with different identifiers.
        missing_envs = np.vstack(np.where(out == 0)).T
        if len(missing_envs) > 0:
            for (atom, radius) in missing_envs:
                env = inci[radius, atom, :]
                radii, atoms = np.where((np.all(inci == env, axis=-1)))
                env_matches = out[(atoms, radii)]
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
        generator (MorganGenerator): Morgan fingerprint generator.
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
    >>> sas = SortAndSlice(molecules, generator, fpsize=128)
    >>> encoded_moleculess = sas(molecules)
    """
    def __init__(
        self,
        generator: MorganGenerator,
        molecules: list[Chem.Mol] = None,
        fpsize: int = None,
        save_img: bool = False,
        central_atom_colors: dict = default_atom_colors,
        ring_color: tuple[float] = (1,1,1),
        aromatic_color: tuple[float] = (1,1,1),
        extra_color: tuple[float] = (0.6, 0.6, 0.6),
        verbose: bool = False,
    ):
        self.generator = generator
        self.verbose = verbose
        self.identifiers = {}
        self.encoder = {}
        self.decoder = {}
        self.fpsize = fpsize
        self.save_img = save_img
        self.atom_colors = central_atom_colors
        self.ring_color = ring_color
        self.aromatic_color = aromatic_color
        self.extra_color = extra_color

        if molecules is not None:
            self.update(molecules)
            self.slice(fpsize)

    def append(self, mol: Chem.Mol|np.ndarray):
        """
        Adds identifiers from a molecule to the identifiers attribute.
        """
        if mol is None:
            return
        elif isinstance(mol, np.ndarray):
            envs = mol
            radius = envs.shape[1] - 1
        elif isinstance(mol, Chem.Mol):
            radius = self.generator.radius
            envs = self.generator.environments(mol)
        else:
            raise ValueError('Input must be a RDKit molecule or an envs array.')

        if self.save_img: 
            bit_info = self.generator.bitinfo(mol)
        else: 
            bit_info = None

        starter = {r: 0 for r in range(radius + 1)}
        starter['num_mols'] = 0
        starter['count'] = 0
        done = {}
        for r in range(radius + 1):
            identifiers, counts = np.unique(envs[:,r], return_counts=True)
            for j in range(len(identifiers)):
                id = int(identifiers[j])
                count = int(counts[j])
                value = self.identifiers.get(id, starter.copy())
                value['count'] += count
                value[r] += count
                if id not in done:
                    value['num_mols'] += 1
                    if self.save_img:
                        atom = mol.GetAtomWithIdx(
                            bit_info[id][0][0]
                        )
                        value['central_atom'] = atom.GetSymbol()
                        value['img'] = Draw.DrawMorganBit(
                            mol, bitId=id, bitInfo=bit_info,
                            extraColor=self.extra_color,
                            centerColor=self.atom_colors.get(value['central_atom'], (1,1,1)),
                            aromaticColor=self.aromatic_color, 
                            ringColor=self.ring_color,
                        )
                        
                        value["aromatic"] = atom.GetIsAromatic()
                        value["ring"] = atom.IsInRing()
                        
                    done[id] = True
                self.identifiers[id] = value

    def update(self, molecules: list[Chem.Mol|np.ndarray]):
        """
        Updates the identifiers attribute with identifiers from new molecules.
        
        Parameters:
        ----------
            molecules (list[Chem.Mol]): List of RDKit molecules.

        Sets:
        ------
            identifiers (dict[str, int]): Dictionary of identifiers and their counts.
        """
        self.pbar = tqdm(total=len(molecules), desc='Collecting identifiers', disable=not self.verbose)
        for mol in molecules:
            self.append(mol)
            self.pbar.update(1)
        self.pbar.close()
        self.pbar = None

    def sort(self, key_order = ('num_mols', 'count')):
        """
        Sorts the identifiers by the number of molecules they appear in and their total count.
        """                
        self.identifiers = dict(sorted(
            self.identifiers.items(), key=lambda x: (x[1][key_order[0]], x[1][key_order[1]]), reverse=True,
        ))

    def slice(self, fpsize: int|None = None):
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
        if fpsize is None:
            if self.fpsize is None:
                self.fpsize = len(self.identifiers)
            fpsize = self.fpsize
        if self.verbose:
            print(f'Attempting to set bit length of encoder to a max of {fpsize}.')
        encoder = {}
        decoder = {}
        count = 0
        for k in self.identifiers.keys():
            if count >= fpsize:
                break
            encoder[k] = count
            decoder[count] = k
            count += 1

        self.encoder = encoder
        self.decoder = decoder
        if len(encoder) < fpsize:
            if self.verbose:
                print(
                    f'Fewer observed substructures than fpsize.\nEncoder is only {len(encoder)} bits long.'
                )

        if self.verbose:
            print(f'Encoder set to {len(encoder)} bits.')

    def encode(self, mol: Chem.Mol|np.ndarray) -> np.ndarray:
        """
        Encodes a molecule into a binary sort and slice vector.

        Parameters:
        ----------
            mol (Chem.Mol): RDKit molecule.

        Returns:
        -------
            np.ndarray: Binary vector indicating substructure presence.
        """
        out = np.zeros(len(self.encoder))
        if mol is None:
            return np.full(len(self.encoder), np.nan)
        elif isinstance(mol, np.ndarray):
            bitmap = np.unique(mol).astype(int)
        elif isinstance(mol, Chem.Mol):
            bitmap = self.generator.bitinfo(mol)
        else:
            raise ValueError('Input must be a RDKit molecule or an envs array.')
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
        if molecules is None:
            return None
        if self.encoder is None:
            print('No encoder found, updating identifiers and slicing.') if self.verbose else None
            self.update(molecules)
            self.slice()
        if isinstance(molecules, Chem.Mol):
            molecules = [molecules]
        self.pbar = tqdm(
            total=len(molecules),
            desc='Encoding molecules',
            disable=not self.verbose
        )
        out = np.zeros((len(molecules), len(self.encoder)))
        for i, mol in enumerate(molecules):
            out[i] = self.encode(mol)
            self.pbar.update(1)
        self.pbar.close()
        self.pbar = None
        return out
    
    def __repr__(self) -> str:
        return f'SortAndSlice(num_envs={len(self.identifiers)}, fpsize={len(self.encoder)})'
    
    def __str__(self) -> str:
        return self.__repr__()
    
    def __getitem__(self, item):
        return self.identifiers[item]
    
    def get(self, key, default=None):
        return self.identifiers.get(key, default)
    
    def items(self):
        return self.identifiers.items()
    
    def keys(self) -> list:
        return self.identifiers.keys()
    
    def values(self) -> list:
        return self.identifiers.values()
    
    def clear(self):
        self.identifiers = {}
        self.encoder = None

class MolDesc:

    def __init__(self, descriptors: list[str], verbose: bool = False, **kwargs):
        self.verbose = verbose
        self.set_generator(descriptors=descriptors)
        self.descriptors = descriptors

    def set_generator(self, descriptors):
        self.generator = MolecularDescriptorCalculator(descriptors)

    def __len__(self):
        return len(self.generator.GetDescriptorNames())

    def __call__(self, X: Chem.Mol|list[Chem.Mol]) -> np.ndarray:
        if isinstance(X, Chem.Mol):
            X = [X]
        out = np.full((len(X), len(self)), np.nan)

        self.pbar = tqdm(
            total=len(X), desc='Calculating Descriptors', disable=not self.verbose
        )
        for idx, mol in enumerate(X):
            if isinstance(mol, Chem.Mol):
                out[idx] = self.generator.CalcDescriptors(mol)
            self.pbar.update()
        self.pbar.close()
        self.pbar = None
        return out

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
        cleanup: bool = True,
        fragment_parent: bool = True,
        neutralize: bool = True,
        reionize: bool = False,
        canonical_tautomer: bool = True,
        keep_chirality: bool = True,
        verbose: bool = False,
        break_at_none: bool = False
    ):
        self.sanitize = sanitize
        self.cleanup = cleanup
        self.fragment_parent = fragment_parent
        self.neutralize = neutralize
        self.reionize = reionize
        self.canonical_tautomer = canonical_tautomer
        self.tautomer_keep_chirality = keep_chirality
        self.verbose = verbose
        self.break_at_none = break_at_none

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
        print('Running standardizer') if self.verbose else None
        try:
            mol.UpdatePropertyCache()
            if self.sanitize:
                mol = self.run_sanitize(mol)

            if self.cleanup:
                mol = self.run_cleanup(mol)

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
        except Exception as e:
            print(f'Error standardizing molecule: {e}')
            return None
        
    def __call__(self, mol: Chem.Mol|list[Chem.Mol]) -> Chem.Mol:
        """
        Standardizes a molecule.

        Parameters:
        ----------
            mol (Chem.Mol): RDKit molecule.

        Returns:
        -------
            Chem.Mol: Standardized RDKit molecule.
        """
        print(f'Standardizer: {self}') if self.verbose else None
        if isinstance(mol, Chem.Mol):
            return self.standardize(mol)
        elif isinstance(mol, list):
            verb = self.verbose
            self.verbose = False
            out = []
            self.pbar = tqdm(mol, disable=not verb, desc='Standardizing molecules')
            for idx, m in enumerate(mol):
                if m is None:
                    print(f'None provide at: {idx}') if verb else None
                    out.append(None)
                    continue
                assert isinstance(m, Chem.Mol), 'Input must be an RDKit molecule.'
                m = self.standardize(m)
                out.append(m)
                self.pbar.update()
                if self.break_at_none and m is None:
                    print(f'Failed at: {idx}') if verb else None
                    break
            self.pbar.close()
            self.pbar = None
            self.verbose = verb
            return out
        else:
            raise ValueError(
                'Input must be a RDKit molecule or a list of RDKit molecules.'
            )
        
    @property
    def settings(self):
        return {
            'sanitize': self.sanitize,
            'cleanup': self.cleanup,
            'fragment_parent': self.fragment_parent,
            'neutralize': self.neutralize,
            'reionize': self.reionize,
            'canonical_tautomer': self.canonical_tautomer,
            'tautomer_keep_chirality': self.tautomer_keep_chirality
        }
    
    def __repr__(self):
        settings = [f'{k}={v}' for k, v in self.settings.items()]
        settings = ', '.join(settings)
        return f'Standardizer({settings})'

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
        Chem.SanitizeMol(mol)
        return mol

    def run_cleanup(self, mol: Chem.Mol) -> Chem.Mol:
        return rdMolStandardize.Cleanup(mol)

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
        return rdMolStandardize.FragmentParent(mol, skipStandardize=False)

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
        Canonicalizes tautomers.

        Can remove chirality: https://github.com/rdkit/rdkit/issues/5531

        Parameters:
        ----------
            mol (Chem.Mol): RDKit molecule.

        Returns:
        -------
            Chem.Mol: Canonicalized RDKit molecule.
        """
        te = rdMolStandardize.TautomerEnumerator()
        if self.tautomer_keep_chirality:
            te.SetRemoveSp3Stereo(False)
        return te.Canonicalize(mol)

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

class FPOps:
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
    
    def pairwise_tanimoto(
        fps: list[DataStructs.ExplicitBitVect],
        verbose: bool = False
    ):
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
        pbar = tqdm(
            total=len(fps), disable=not verbose,
            desc='Calculating Tanimoto similarities'
        )
        for i in range(len(fps)):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
            similarities[i, :i], similarities[:i, i] = sims, sims
            pbar.update(1)
        pbar.close()
        return similarities
    
    def butina(
        fps: list[DataStructs.ExplicitBitVect],
        threshold: float = 0.65,
        verbose: bool = False,
    ):
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
        distances = 1 - FPOps.pairwise_tanimoto(fps, verbose=verbose)
        distances = distances[np.tril_indices(len(distances), -1)]
        print('Calculating clusters...') if verbose else None
        clusters = Butina.ClusterData(distances, distThresh=threshold, isDistData=True, nPts=len(fps))
        
        molecule_clusters = np.zeros(len(fps), dtype=int)
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
            radius = 1: Identity matrix, Adjacency matrix.
            radius = 2: Identity matrix, Adjacency matrix, Adjacency matrix + 2nd degree neighbors.
            radius = -1: Continues until out[-1, :, :] is all ones.
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
