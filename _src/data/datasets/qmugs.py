from .base import BaseDataset
from ..mol import Standardizer
from pathlib import Path
import pandas as pd
from tqdm import tqdm

class QMugs(BaseDataset):
    """
    Pandas DataFrame class for QMugs dataset.

    QMugs is a dataset of quantum mechanical properties of drug-like molecules.
    Some SMILES strings in QMugs csv have incorrect stereochemistry; to account for this
    canonical SMILES are also obtained directly from CHEMBL v.27 using CHEMBL IDs in QMugs csv.

    Paper:
        Isert, C., Atz, K., Jiménez-Luna, J. et al.
        QMugs, quantum mechanical properties of drug-like molecules.
        Sci Data 9, 273 (2022). https://doi.org/10.1038/s41597-022-01390-7

    Parameters
    ----------
    root: str
        The root directory to store the dataset.
    compression: bool
        Whether to compress the dataset or not.
        Default is True.
    
    Attributes
    ----------
    root: str
        The root directory to store the dataset.
    csv: str
        The path to the dataset.
    """

    url = "https://libdrive.ethz.ch/index.php/s/X5vOBNSITAG5vzM/download?path=%2F&files=summary.csv"

    def __init__(
        self, root: str|None = None, compression: bool = True,
        verbose: bool = True, standardizer: Standardizer = Standardizer()
    ):
        """
        Initialize the QMugs dataset.
        """
        # Set the suffix and compression
        suffix = 'csv.gz' if compression else 'csv'

        # Set the path to the csv file
        csv = Path(root) / f'qmugs.{suffix}'
        # Initialize the BaseDataset
        super(QMugs, self).__init__(csv=csv, url=self.url, compression=compression, verbose=verbose, standardizer=standardizer)

        # Set the root directory and csv file
        self.root = root
        self.csv = csv

        # obtain canonical smiles from CHEMBL v.27
        if 'SMILES' not in self.columns:
            # Drop all columns except 'chembl_id' and 'smiles'
            self.drop(
                self.columns.difference(['chembl_id', 'smiles']),
                axis=1, inplace=True
            )
            
            # Keep only one conformer row for each molecule
            self.drop_duplicates(subset='chembl_id', inplace=True)

            # Download CHEMBL v.27 chemreps file
            chembl_url = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_27/chembl_27_chemreps.txt.gz"
            chemble_v27 = pd.read_csv(chembl_url, sep='\t')

            # Map CHEMBL IDs to canonical SMILES
            self['SMILES'] = self['chembl_id'].map(chemble_v27.set_index('chembl_id')['canonical_smiles'])

            # Drop rows where canonical SMILES are duplicates.
            # This is to account for cases where multiple CHEMBL IDs map to the same
            # canonical SMILES.
            # The inchi keys are different; when read into RDKit the molecules are the same
            # Differences between inchi keys are due to stereocenters being identified in one
            # molecule and not in the other; type of stereochemistry is not specified,
            # hence why the molecules are the same.
            # 32 molecules are removed by this step.
            self.drop_duplicates(subset='SMILES', inplace=True)
            self.drop(
                self.columns.difference(['chembl_id', 'SMILES']),
                axis=1, inplace=True
            )
            self.mol_standardize_check()
            # Reset the index and save the dataset
            self.reset_index(drop=True, inplace=True)
            self.to_csv(csv, index=False, compression=compression)

    