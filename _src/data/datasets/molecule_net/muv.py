from ..base import BaseDataset

from pathlib import Path

class MUV(BaseDataset):
    """
    Maximum Unbiased Validation dataset.

    The Maximum Unbiased Validation (MUV) dataset is a collection of 17 assay results from
    PCBA, with filters for noisy/incorrect data. Each task is a binary classificaiton of
    activity against a particular target. 

    This is better than the PCBA dataset as filters for erroneous data are applied and assays
    are separated into different tasks rather than aggrgated. In DeepChem, the 17 tasks are
    aggregated into a single dataset, with NaN values being replaced with 0. This should be
    avoided, as each SMILES with a NaN value has not been assessed on that target; presuming
    inactivity without a known experimental result will introduce noise. 
    """
    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/muv.csv.gz'

    def __init__(self, root: str|None = None, compression: bool = True):
        suffix = 'csv' if compression else 'csv.gz'
        csv = Path(root) / f'muv.{suffix}' if root else None
        super(MUV, self).__init__(csv=csv, url=self.url, compression=compression)
        self.rename(
            columns={'smiles': 'SMILES'},
            inplace=True
        )
        self.save()

class MUV466(MUV):
    """
    MUV dataset for PCBA assay AID 466; S1P1 Agonists. 

    https://pubchem.ncbi.nlm.nih.gov/bioassay/466

    This is a subset of the MUV dataset, with only the S1P1 Agonists assay.
    """
    def __init__(self, root: str|None = None, compression: bool = True):
        super(MUV466, self).__init__(root=root, compression=compression)
        self.rename(
            columns={'MUV-466': 'y'},
            inplace=True
        )
        self.drop(
            self.columns.difference(['SMILES', 'y']),
            axis=1, inplace=True
        )
        self.dropna(subset=['y'], inplace=True)

class MUV548(MUV):
    """
    MUV dataset for PCBA assay AID 548; protein kinase A (PKA) inhibitors.

    https://pubchem.ncbi.nlm.nih.gov/bioassay/548

    This is a subset of the MUV dataset, with only the PKA inhibitors assay.
    """
    def __init__(self, root: str|None = None, compression: bool = True):
        super(MUV548, self).__init__(root=root, compression=compression)
        self.rename(
            columns={'MUV-548': 'y'},
            inplace=True
        )
        self.drop(
            self.columns.difference(['SMILES', 'y']),
            axis=1, inplace=True
        )
        self.dropna(subset=['y'], inplace=True)

class MUV600(MUV):
    """
    MUV dataset for PCBA assay AID 600; nuclear receptor Steroidogenic Factor 1 (SF-1) inhibitors.

    https://pubchem.ncbi.nlm.nih.gov/bioassay/600

    This is a subset of the MUV dataset, with only the SF-1 inhibitors assay.
    """
    def __init__(self, root: str|None = None, compression: bool = True):
        super(MUV600, self).__init__(root=root, compression=compression)
        self.rename(
            columns={'MUV-600': 'y'},
            inplace=True
        )
        self.drop(
            self.columns.difference(['SMILES', 'y']),
            axis=1, inplace=True
        )
        self.dropna(subset=['y'], inplace=True)

class MUV644(MUV):
    """
    MUV dataset for PCBA assay AID 644; Rho kinase 2 (Rock2) inhibitors.

    https://pubchem.ncbi.nlm.nih.gov/bioassay/644

    This is a subset of the MUV dataset, with only the Rock2 inhibitors assay.
    """
    def __init__(self, root: str|None = None, compression: bool = True):
        super(MUV644, self).__init__(root=root, compression=compression)
        self.rename(
            columns={'MUV-644': 'y'},
            inplace=True
        )
        self.drop(
            self.columns.difference(['SMILES', 'y']),
            axis=1, inplace=True
        )
        self.dropna(subset=['y'], inplace=True)

class MUV652(MUV):
    """
    MUV dataset for PCBA assay AID 652; HIV-1 RT-RNase H inhibitors.

    https://pubchem.ncbi.nlm.nih.gov/bioassay/652

    This is a subset of the MUV dataset, with only the RT-RNH inhibitors assay.
    """
    def __init__(self, root: str|None = None, compression: bool = True):
        super(MUV652, self).__init__(root=root, compression=compression)
        self.rename(
            columns={'MUV-652': 'y'},
            inplace=True
        )
        self.drop(
            self.columns.difference(['SMILES', 'y']),
            axis=1, inplace=True
        )
        self.dropna(subset=['y'], inplace=True)

class MUV689(MUV):
    """
    MUV dataset for PCBA assay AID 689; ephrin type-A receptor 4 precursor antagonists.

    https://pubchem.ncbi.nlm.nih.gov/bioassay/689

    This is a subset of the MUV dataset, with only the EphA4 receptor antagonists assay.
    """
    def __init__(self, root: str|None = None, compression: bool = True):
        super(MUV689, self).__init__(root=root, compression=compression)
        self.rename(
            columns={'MUV-689': 'y'},
            inplace=True
        )
        self.drop(
            self.columns.difference(['SMILES', 'y']),
            axis=1, inplace=True
        )
        self.dropna(subset=['y'], inplace=True)

class MUV692(MUV):
    """
    MUV dataset for PCBA assay AID 692; nuclear receptor Steroidogenic Factor 1 (SF-1) activators.

    https://pubchem.ncbi.nlm.nih.gov/bioassay/692

    This is a subset of the MUV dataset, with only the SF-1 activators assay.
    """
    def __init__(self, root: str|None = None, compression: bool = True):
        super(MUV692, self).__init__(root=root, compression=compression)
        self.rename(
            columns={'MUV-692': 'y'},
            inplace=True
        )
        self.drop(
            self.columns.difference(['SMILES', 'y']),
            axis=1, inplace=True
        )
        self.dropna(subset=['y'], inplace=True)

class MUV712(MUV):
    """
    MUV dataset for PCBA assay AID 712; tumor Hsp90 inhibitors.

    https://pubchem.ncbi.nlm.nih.gov/bioassay/712

    This is a subset of the MUV dataset, with only the Hsp90 inhibitors assay.
    """
    def __init__(self, root: str|None = None, compression: bool = True):
        super(MUV712, self).__init__(root=root, compression=compression)
        self.rename(
            columns={'MUV-712': 'y'},
            inplace=True
        )
        self.drop(
            self.columns.difference(['SMILES', 'y']),
            axis=1, inplace=True
        )
        self.dropna(subset=['y'], inplace=True)

class MUV713(MUV):
    """
    MUV dataset for PCBA assay AID 713; Estrogen Receptor-alpha Coactivator Binding Inhibitors

    https://pubchem.ncbi.nlm.nih.gov/bioassay/713

    This is a subset of the MUV dataset, with only the ER-alpha inhibitors assay.
    """
    def __init__(self, root: str|None = None, compression: bool = True):
        super(MUV713, self).__init__(root=root, compression=compression)
        self.rename(
            columns={'MUV-713': 'y'},
            inplace=True
        )
        self.drop(
            self.columns.difference(['SMILES', 'y']),
            axis=1, inplace=True
        )
        self.dropna(subset=['y'], inplace=True)

class MUV733(MUV):
    """
    MUV dataset for PCBA assay AID 733; Estrogen Receptor-beta Coactivator Binding Inhibitors

    https://pubchem.ncbi.nlm.nih.gov/bioassay/733

    This is a subset of the MUV dataset, with only the ER-beta inhibitors assay.
    """
    def __init__(self, root: str|None = None, compression: bool = True):
        super(MUV733, self).__init__(root=root, compression=compression)
        self.rename(
            columns={'MUV-733': 'y'},
            inplace=True
        )
        self.drop(
            self.columns.difference(['SMILES', 'y']),
            axis=1, inplace=True
        )
        self.dropna(subset=['y'], inplace=True)

class MUV737(MUV):
    """
    MUV dataset for PCBA assay AID 737; Estrogen Receptor-alpha Coactivator Binding Potentiators.

    https://pubchem.ncbi.nlm.nih.gov/bioassay/737

    This is a subset of the MUV dataset, with only the ER-alpha potentiators assay.
    """
    def __init__(self, root: str|None = None, compression: bool = True):
        super(MUV737, self).__init__(root=root, compression=compression)
        self.rename(
            columns={'MUV-737': 'y'},
            inplace=True
        )
        self.drop(
            self.columns.difference(['SMILES', 'y']),
            axis=1, inplace=True
        )
        self.dropna(subset=['y'], inplace=True)

class MUV810(MUV):
    """
    MUV dataset for PCBA assay AID 810; Focal Adhesion Kinase (FAK) inhibitors.

    https://pubchem.ncbi.nlm.nih.gov/bioassay/810

    This is a subset of the MUV dataset, with only the FAK inhibitors assay.
    """
    def __init__(self, root: str|None = None, compression: bool = True):
        super(MUV810, self).__init__(root=root, compression=compression)
        self.rename(
            columns={'MUV-810': 'y'},
            inplace=True
        )
        self.drop(
            self.columns.difference(['SMILES', 'y']),
            axis=1, inplace=True
        )
        self.dropna(subset=['y'], inplace=True)

class MUV832(MUV):
    """
    MUV dataset for PCBA assay AID 832; Cathepsin G inhibitors.

    https://pubchem.ncbi.nlm.nih.gov/bioassay/832

    This is a subset of the MUV dataset, with only the Cathepsin G inhibitors assay.
    """
    def __init__(self, root: str|None = None, compression: bool = True):
        super(MUV832, self).__init__(root=root, compression=compression)
        self.rename(
            columns={'MUV-832': 'y'},
            inplace=True
        )
        self.drop(
            self.columns.difference(['SMILES', 'y']),
            axis=1, inplace=True
        )
        self.dropna(subset=['y'], inplace=True)

class MUV846(MUV):
    """
    MUV dataset for PCBA assay AID 846; Factor XIa inhibitors.

    https://pubchem.ncbi.nlm.nih.gov/bioassay/846

    This is a subset of the MUV dataset, with only the Factor XIa inhibitors assay.
    """
    def __init__(self, root: str|None = None, compression: bool = True):
        super(MUV846, self).__init__(root=root, compression=compression)
        self.rename(
            columns={'MUV-846': 'y'},
            inplace=True
        )
        self.drop(
            self.columns.difference(['SMILES', 'y']),
            axis=1, inplace=True
        )
        self.dropna(subset=['y'], inplace=True)

class MUV852(MUV):
    """
    MUV dataset for PCBA assay AID 852; Factor XIIa inhibitors.

    https://pubchem.ncbi.nlm.nih.gov/bioassay/852

    This is a subset of the MUV dataset, with only the Factor XIIa inhibitors assay.
    """
    def __init__(self, root: str|None = None, compression: bool = True):
        super(MUV852, self).__init__(root=root, compression=compression)
        self.rename(
            columns={'MUV-852': 'y'},
            inplace=True
        )
        self.drop(
            self.columns.difference(['SMILES', 'y']),
            axis=1, inplace=True
        )
        self.dropna(subset=['y'], inplace=True)

class MUV858(MUV):
    """
    MUV dataset for PCBA assay AID 858; DRD1 allosteric modulators.

    https://pubchem.ncbi.nlm.nih.gov/bioassay/858

    This is a subset of the MUV dataset, with only the DRD1 allosteric modulators assay.
    """
    def __init__(self, root: str|None = None, compression: bool = True):
        super(MUV858, self).__init__(root=root, compression=compression)
        self.rename(
            columns={'MUV-858': 'y'},
            inplace=True
        )
        self.drop(
            self.columns.difference(['SMILES', 'y']),
            axis=1, inplace=True
        )
        self.dropna(subset=['y'], inplace=True)

class MUV859(MUV):
    """
    MUV dataset for PCBA assay AID 859; M1 muscarinic acetylcholine receptor antagonists.

    https://pubchem.ncbi.nlm.nih.gov/bioassay/859

    This is a subset of the MUV dataset, with only the mAChR antagonists assay.
    """
    def __init__(self, root: str|None = None, compression: bool = True):
        super(MUV859, self).__init__(root=root, compression=compression)
        self.rename(
            columns={'MUV-859': 'y'},
            inplace=True
        )
        self.drop(
            self.columns.difference(['SMILES', 'y']),
            axis=1, inplace=True
        )
        self.dropna(subset=['y'], inplace=True)
