import torch
import torch_geometric as pyg
from typing import Any, List
from _src.data.datasets import GraphDataset

class Collater(pyg.loader.dataloader.Collater):
    """
    Collater for the GraphDataset. Same as the default PyG collater,
    but also adds a global index to the batch.

    The global index is a tensor that contains the indexes for the
    global node embeddings for each graph in the batch.
    Used for the global node embedding in the GraphDataset;
    hyperparameter option for graph pooling type.
    """
    def __call__(self, batch: List[Any]) -> Any:
        batch = super(Collater, self).__call__(batch)
        if 'global_idx' in batch:
            batch.global_idx[0] -= 1
            batch.global_idx += 1
            batch.global_idx = batch.global_idx.cumsum(0)
        return batch
    
class DataLoader(torch.utils.data.DataLoader):
    """
    DataLoader for the GraphDataset. Same as the default PyG DataLoader,
    but uses the custom Collater for handling global graph nodes.
    """
    def __init__(
        self,
        dataset: GraphDataset,
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: List[str]|None = None,
        exclude_keys: List[str]|None = None,
        **kwargs,
    ):
        # Remove for PyTorch Lightning:
        kwargs.pop('collate_fn', None)

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=Collater(dataset, follow_batch, exclude_keys),
            **kwargs,
        )