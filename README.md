# topological-pretraining

This repository contains the code for "Improving Graph Neural Networks for QSAR via pre-training on Extended-Connectivity Fingerprints".

## Installing within conda env

```shell
git clone git@github.com:oxpig/topological-pretraining.git
cd topological-pretraining
conda env create --file environment.yaml
conda activate topological-pretraining
```

## Notebooks

See:
    - [Quickstart notebook](./notebooks/00_quickstart.ipynb) for basic usage.
    - [Dataset notebook](./notebooks/01_datasets.ipynb) for loading and setting up datasets.
    - [Models notebook](./notebooks/02_models.ipynb) for basic model demonstrations.
    - [Benchmark evaluations](./notebooks/03_benchmark_evaluation.ipynb) for statistical tests and visualisations.
    - [Substructure importance notebook](./notebooks/04_substructure_importance.ipynb) for estimating pre-trained GIN substructure importance.
