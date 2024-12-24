from .bace import BACE
from .bbbp import BBBP
from .clintox import ClinTox
from .esol import ESOL
from .freesolv import FreeSolv
from .hiv import HIV
from .lipo import Lipo
from .molecule_net import MoleculeNet
from .muv import (
    MUV, MUV466, MUV548,
    MUV600, MUV644, MUV652,
    MUV689, MUV692, MUV712,
    MUV713, MUV733, MUV737,
    MUV810, MUV832, MUV846,
    MUV852, MUV858, MUV859,
)
from .sider import SIDER
from .tox21 import Tox21
from .toxcast import ToxCast

"""
See: https://practicalcheminformatics.blogspot.com/2023/08/we-need-better-benchmarks-for-machine.html
Excluded datasets:
-   QM7, QM7b, QM8, QM9. Endpoints are not relevant to 2D molecular representations as they are
    conformer-dependent.
-   PCBA. Binding endpoints are intertarget between assays; prediction without a representation
    of the target is meaningless, as predictions for a ligand will be identical regardless of the
    target pathway/gene/protein/function/cell type. Active endpoints also mean a combination of
    inhibitory, agonistic, and cytotoxic effects; this conflation makes little sense as
    they are very different. Additionally, active endpoints vary in potency, adding further noise.
-   PDBbind. Endpoints are binding affinities, which are dependent on the target protein. Not useful
    for benchmarking ligand representations alone.

Unused datasets:
-   BACE. Due to unmarked stereochemistry in SMILES strings. Endpoints are also IC50 values from
    multiple publications. IC50 values are dependent on specific assay concentrations;
    aggregation of IC50 values introduces noise.
-   BBBP. Higly hetreogeneous dataset variable definitions for the endpoint, likely to be noisy and
    not very informative.
-   HIV. ~70% of the dataset trigger structural alerts; risk of noise, false positives, and ligands
    that are not druglike.
-   SIDER. Endpoints are derived from qualitative data and is therefore unlikely to be robust as a
    benchmark.
-   ToxCast. High proportion of molecules contain CHEMBL structural alerts; risk of noise.
    Additionally, endpoints are from cellular rather than biophysical assays; risk of noise.
-   ClinTox. Qualitative toxicity endpoints, not very specitic.

Used datasets:
-   MUV. Risk of overfitting, but should be reduced by Butina split. Also derived from PCBA, but
    is separated into individual assays each for a specific target. For each assay, each active
    label should consistently represent the same type of activity.
-   Lipo (Lipophilicity).
-   Tox21. (Endpoints are from cell rather than biophysical assays; risk of noise.)
-   FreeSolv. Not particularly relevant to drug discovery but still useful for indicating
    generalization to multiple endpoints.
-   ESOL. Dataset range is larger than is typical for drug discovery. Can get around this by
    also evaluating on a subset of the test set only within a specific range.
"""
