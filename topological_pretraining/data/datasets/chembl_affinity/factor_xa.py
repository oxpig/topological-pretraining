from pathlib import Path

import numpy as np

from ...mol import Standardizer
from ..base import BaseDataFrame


class FactorXA(BaseDataFrame):
    """
    Dataset of binding affinities of small molecules with Factor XA.
    Data from ChEMBL, curated by Dablander, M. et al. (2023)
    (https://doi.org/10.1186/s13321-023-00708-w)
    """

    url = "https://raw.githubusercontent.com/MarkusFerdinandDablander/QSAR-activity-cliff-experiments/refs/heads/main/data/chembl_factor_xa/molecule_data_clean.csv"

    def __init__(
        self,
        root: str | None = None,
        compression: bool = True,
        verbose: bool = False,
        standardizer: Standardizer = Standardizer(),
    ):
        suffix = "csv.gz" if compression else "csv"
        csv = Path(root) / f"factor_xa.{suffix}" if root else None
        super().__init__(
            csv=csv,
            url=self.url,
            compression=compression,
            verbose=verbose,
            standardizer=standardizer,
        )
        if "y" not in self.columns:
            self.rename(
                columns={"Ki [nM]": "y"},
                inplace=True,
            )
            self.y = self.y.apply(lambda x: -np.log10(x) + 9)
            self.mol_standardize_check()
            self.save()

    @property
    def task(self):
        return "regression"
