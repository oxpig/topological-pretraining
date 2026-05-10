from .esol import ESOL
from .freesolv import FreeSolv
from .lipo import Lipo
from .muv import MUV
from .tox21 import Tox21


class MoleculeNet:
	"""
	Molecule Net datasets.
	Wu, Z. et al. (2018).
	(https://arxiv.org/abs/1703.00564)

	Details on reasons for inclusion and exclusion of particular datasets from MoleculeNet
	in this study are below, along with cautionary notes on the included datasets.

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
	    is separated into individual assays, each for a specific target. For each assay, each active
	    label should consistently represent the same type of activity. Effectively, MUV is 17 separate
	    datasets, with similar but non-identical binary classification tasks. Each assay has ~15000
	    decoy ligands (labelled 0) and ~30 actives (labelled 1). This ratio of actives to decoys is
	    useful, as it reflects the sparcity of real-world drug discovery screening. Assays SHOULD NOT
	    be aggregated into one multi-classification task, as NaN values in the dataset != inactive;
	    NaN == not necessarily tested. Aggregation increases the risk of false negatives which is
	    detrimental accurate benchmarking, particulary given how sparse the dataset is!
	-   Lipo (Lipophilicity).
	-   Tox21. (Endpoints are from cell rather than biophysical assays; risk of noise.)
	-   FreeSolv. Not particularly relevant to drug discovery but still useful for indicating
	    generalization to multiple endpoints.
	-   ESOL. Dataset range is larger than is typical for drug discovery. Can get around this by
	    also evaluating on a subset of the test set only within a specific range.
	"""

	def __init__(self, root: str, compression: bool = True):
		self.root = root
		self.suffix = 'csv.gz' if compression else 'csv'
		self.compression = compression

	@property
	def ESOL(self):
		return ESOL(root=self.root, compression=self.compression)

	@property
	def FreeSolv(self):
		return FreeSolv(root=self.root, compression=self.compression)

	@property
	def Lipo(self):
		return Lipo(root=self.root, compression=self.compression)

	@property
	def MUV(self):
		return MUV(root=self.root, compression=self.compression)

	@property
	def Tox21(self):
		return Tox21(root=self.root, compression=self.compression)
