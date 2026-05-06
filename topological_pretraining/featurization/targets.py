from copy import deepcopy

import torch
import torch_geometric as pyg
from rdkit import Chem

from topological_pretraining.featurization import ECFP, FCFP, PDV, SNS
from topological_pretraining.featurization.load_featurizers import read_from_dict

possible_targets_ = {
    "SNS": {
        "head_type": "binary",
        "target_class": SNS,
        "refit": False,
        "level": "global",
        "input_type": "molecule",
        "split_dependent": True,
        "run_dependent": False,
    },
    "ECFP": {
        "head_type": "binary",
        "target_class": ECFP,
        "processing": [],
        "refit": False,
        "level": "global",
        "input_type": "molecule",
        "split_dependent": False,
        "run_dependent": False,
    },
    "FCFP": {
        "head_type": "binary",
        "target_class": FCFP,
        "level": "global",
        "input_type": "molecule",
        "split_dependent": False,
        "run_dependent": False,
    },
    "PDV": {
        "head_type": "regression",
        "target_class": PDV,
        "level": "global",
        "input_type": "molecule",
        "split_dependent": False,
        "run_dependent": False,
    },
}


class Targets(dict):
    """
    Class for turning featurization methods into a dict of targets for pre-training.
    Used to label molecular graphs with various target labels.

    Parameters
    ----------
    targets : dict[str, dict[str, str]]
        A dictionary where keys are target names and values are
        dictionaries of parameters for each target.
        Targets can include 'SNS', 'ECFP', 'FCFP', and 'PDV'.
        For parameters, refer to the individual target classes in the
        `featurization` module.
    targets_path : str, optional
        Path to a file containing saved targets. If provided, the targets will be loaded from this path.
    """

    def __init__(
        self, targets: dict[str, dict[str, str]] = {}, targets_path: str = None
    ):
        targets = deepcopy(targets)
        super().__init__(**targets)
        self.targets_path = targets_path
        if self.targets_path is not None:
            self.load()
        else:
            for target_name, target_kwargs in self.items():
                if target_name not in possible_targets_:
                    raise ValueError(f"Target {target_name} not supported.")
                self[target_name]["pipeline"] = possible_targets_[target_name][
                    "target_class"
                ](**target_kwargs)
                self[target_name]["prediction_head"] = possible_targets_[target_name][
                    "head_type"
                ]
                self[target_name]["level"] = possible_targets_[target_name]["level"]
                self[target_name]["input_type"] = possible_targets_[target_name][
                    "input_type"
                ]

    def fit(self, data: tuple[list[Chem.Mol], list[pyg.data.Data]]):
        """
        Fit the featurizers for each target to a list of molecules and graphs.

        Parameters
        ----------
        data : tuple[list[Chem.Mol], list[pyg.data.Data]]
            A tuple containing a list of RDKit molecule objects and a list of corresponding PyG Data objects.

        Returns
        -------
        Targets
            The instance of Targets with fitted pipelines for each target.
        """
        for target_name in self:
            input_type = self[target_name]["input_type"]

            if input_type == "molecule":
                x = torch.tensor(
                    self[target_name]["pipeline"].fit_transform(data[0]),
                    dtype=torch.float64,
                )
            elif input_type == "graph":
                x = torch.tensor(
                    self[target_name]["pipeline"].fit_transform(data[1]),
                    dtype=torch.float64,
                )
            if self[target_name]["prediction_head"] == "binary":
                zero_class = torch.zeros_like(x, dtype=torch.float64)
                zero_class[x == 0] = 1
                weights = torch.stack(
                    [
                        x.size(0) / (2 * zero_class.sum(dim=0)),
                        x.size(0) / (2 * x.sum(dim=0)),
                    ]
                )
                weights.type(torch.float64)

            elif self[target_name]["prediction_head"] == "multiclass":
                weights = torch.tensor(x.size(0) / (x.size(1) * x.sum(dim=0)))
                weights.type(torch.float64)

            else:
                weights = None
            self[target_name]["class_weights"] = weights

        return self

    def transform(self, mol: Chem.Mol, graph: pyg.data.Data):
        """
        For each target, compute and add the target label to the graph.

        Parameters
        ----------
        mol : Chem.Mol
            The RDKit molecule object to transform.
        graph : pyg.data.Data
            The PyG Data object to transform.

        Returns
        -------
        pyg.data.Data
            The transformed PyG Data object with target labels.
        """

        for target_name in self:
            if target_name in graph:
                continue
            input_type = self[target_name]["input_type"]
            level = self[target_name]["level"]
            prediction_head = self[target_name]["prediction_head"]
            if prediction_head == "regression":
                dtype = torch.float32
            else:
                dtype = torch.long

            if input_type == "molecule":
                X = mol
            else:
                X = graph
            y = self[target_name]["pipeline"].transform(X)

            if level == "global":
                graph[target_name] = torch.tensor(y, dtype=dtype)
            elif level == "node":
                y, mask = y
                graph[target_name] = torch.tensor(y, dtype=dtype)
                graph[f"{target_name}_mask"] = mask
            else:
                raise ValueError(f"Level {level} not supported.")

        return graph

    def to_dict(self):
        """
        Convert the Targets instance to a dictionary format.

        Returns:
        -------
        dict
            A dictionary representation of the Targets instance,
            where each target's pipeline is converted to a dictionary
        """
        out = {}
        for target in self:
            out[target] = self[target].copy()
            out[target]["pipeline"] = self[target]["pipeline"].to_dict()
        return out

    def save(self, targets_path: str = None):
        """
        Save the Targets instance to a file.

        Parameters:
        ----------
        targets_path : str, optional
            The file path where the targets will be saved. If not provided, the existing targets_path
            will be used. If no path is set, the targets will not be saved.
        """
        if targets_path is None:
            targets_path = self.targets_path
        else:
            self.targets_path = targets_path
        torch.save(self.to_dict(), targets_path)

    def load(self):
        """
        Load the Targets instance from a file.

        If the targets_path is set, it will load the targets from that path.
        If the targets_path is not set, it will not load any targets.
        """
        targets = torch.load(self.targets_path, weights_only=True)
        for target in targets:
            self[target] = targets[target]
            self[target]["pipeline"] = read_from_dict(self[target]["pipeline"])
            self[target]["prediction_head"] = possible_targets_[target]["head_type"]
            self[target]["level"] = possible_targets_[target]["level"]
            self[target]["input_type"] = possible_targets_[target]["input_type"]

    @property
    def is_fitted_(self):
        """
        Check if all pipelines in the Targets instance are fitted.

        Returns
        -------
        bool
            True if all pipelines are fitted, False otherwise.
        """
        return all(
            [
                self[target_name]["pipeline"].__sklearn_is_fitted__()
                for target_name in self
            ]
        )
