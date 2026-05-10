import traceback
from collections.abc import Callable

import numpy as np
import torch
import torch_geometric as pyg
from joblib import Parallel, delayed
from sklearn.inspection._permutation_importance import (
	_create_importances_bunch,
	_weights_scorer,
)
from sklearn.metrics import check_scoring
from sklearn.metrics._scorer import _MultimetricScorer
from sklearn.model_selection._validation import _aggregate_score_dicts

from topological_pretraining.featurization.pretrained import PreTrainedGNN


# graph batching
def batch_graphs(graphs: list[pyg.data.Data], batch_size: int | None = None):
	"""
	Batches a list of PyTorch Geometric Data objects into a single batch.

	Parameters:
	----------
	graphs : list[pyg.data.Data]
	    A list of PyTorch Geometric Data objects representing individual graphs.

	Returns:
	-------
	pyg.data.Batch
	    A PyTorch Geometric Batch object containing all graphs in the input list.
	"""
	if batch_size is not None:
		batches = [
			pyg.data.Batch.from_data_list(graphs[i : i + batch_size])
			for i in range(0, len(graphs), batch_size)
		]
		return batches
	else:
		return pyg.data.Batch.from_data_list(graphs)


# get node permutation indexes
def get_permutations(dim, repeats=5, random_state=42):
	"""
	Generates random permutations of node feature indices.

	Parameters:
	----------
	dim : int
	    The dimension of the node features (number of features for each node).
	"""
	out = torch.zeros((repeats, dim), dtype=torch.int64)
	generator = torch.Generator().manual_seed(random_state)
	for i in range(repeats):
		out[i] = torch.randperm(dim, generator=generator)
	return out


# permute embeddings occurrences of a token
def permute_graph(graph, token_idx, random_idx, perm_type='dist', seed=42):
	"""
	Permutes the embeddings of a specific token in the graph.

	Parameters:
	----------
	graph : pyg.data.Data
	    The graph data object containing node features and tokens.
	token_idx : int
	    The index of the token whose embeddings are to be permuted.
	random_idx : torch.Tensor
	    A tensor containing the random indices to permute the embeddings.
	perm_type : str, optional
	    The type of permutation to apply. Options are "single" for single token permutation
	    and "dist" for distributed permutation across multiple tokens (default is "dist").
	    The "single" type permutes at max one token at a time per graph,
	    while "dist" permutes across all tokens by distributing the random indices over the graph.

	Returns:
	-------
	pyg.data.Data
	    A new graph data object with the permuted embeddings for the specified token.
	"""
	token_mask = graph.tokens == token_idx
	token_indices = torch.nonzero(token_mask, as_tuple=False)

	if token_indices.dim() > 0 and token_indices.size(0) > 0:
		if perm_type == 'single':
			_apply_single_permutation(graph, token_indices, random_idx, seed)
		elif perm_type == 'dist':
			_apply_distributed_permutation(graph, token_indices, random_idx, seed)
	return graph


def _apply_single_permutation(graph, token_indices, random_idx, seed=42):
	"""
	Applies a single permutation to the embeddings of a randomly selected token.

	Parameters:
	----------
	graph : pyg.data.Data
	    The graph data object containing node features and tokens.
	token_indices : torch.Tensor
	    A tensor containing the indices of the tokens to permute.
	random_idx : torch.Tensor
	    A tensor containing the random indices to permute the embeddings.

	Returns:
	-------
	None
	    The graph data object is modified in place with the permuted embeddings.
	"""
	generator = torch.Generator().manual_seed(seed)
	selected = token_indices[
		torch.randint(0, len(token_indices), size=(1,), generator=generator).item()
	]
	graph.x[*selected, :] = graph.x[*selected, random_idx]


def _apply_distributed_permutation(graph, token_indices, random_idx, seed=42):
	"""
	Applies a distributed permutation to the embeddings of multiple tokens.

	Parameters:
	----------
	graph : pyg.data.Data
	    The graph data object containing node features and tokens.
	token_indices : torch.Tensor
	    A tensor containing the indices of the tokens to permute.
	random_idx : torch.Tensor
	    A tensor containing the random indices to permute the embeddings.

	Returns:
	-------
	None
	    The graph data object is modified in place with the permuted embeddings.
	"""
	generator = torch.Generator().manual_seed(seed)
	n_chunks = len(token_indices)
	token_indices = token_indices[
		torch.randperm(token_indices.size(0), generator=generator)
	]
	random_idx_chunks = _chunk_random_indices(random_idx, n_chunks)

	start = 0
	for i in range(len(token_indices)):
		chunk = random_idx_chunks[i]
		graph.x[*token_indices[i], start : start + len(chunk)] = graph.x[
			*token_indices[i], chunk
		]
		start += len(chunk)


def _chunk_random_indices(random_idx, n_chunks):
	"""
	Chunks the random indices into the specified number of chunks.

	Parameters:
	----------
	random_idx : torch.Tensor
	    A tensor containing the random indices to be chunked.
	n_chunks : int
	    The number of chunks to create from the random indices.

	Returns:
	-------
	tuple
	    A tuple containing the chunks of random indices.

	Raises:
	------
	ValueError
	    If the number of chunks cannot be created from the random indices.
	"""
	random_idx_chunks = torch.chunk(random_idx, n_chunks)
	chunk_count = 1

	while len(random_idx_chunks) < n_chunks:
		random_idx_chunks = torch.chunk(random_idx, n_chunks + chunk_count)
		chunk_count += 1
		if len(random_idx_chunks) == n_chunks:
			break
		elif len(random_idx_chunks) < n_chunks:
			continue
		num_to_merge = len(random_idx_chunks) - n_chunks
		new_chunks = [
			random_idx_chunks[i] for i in range(len(random_idx_chunks) - num_to_merge)
		]
		to_merge = [new_chunks[-1]] + list(random_idx_chunks[-num_to_merge:])
		new_chunks[-1] = torch.cat(to_merge)
		random_idx_chunks = tuple(new_chunks)

	if len(random_idx_chunks) != n_chunks:
		raise ValueError(
			f'Could not create {n_chunks} chunks from random_idx of size {len(random_idx)}.'
		)

	return random_idx_chunks


# check whether token appears in list of graphs
def check_token_presence(graph_list, token_idx):
	batch = batch_graphs(graph_list)
	if 'tokens' in batch:
		return torch.any(batch.tokens == token_idx)
	else:
		return torch.any(batch.x == token_idx)


def calculate_token_scores(
	estimator,
	gnn: PreTrainedGNN,
	X: list[pyg.data.Data],
	y: np.ndarray,
	token_idx: int,
	n_repeats: int,
	baseline_score: float | dict,
	random_state: int = 42,
	scorer: str | Callable = '',
	perm_type: str = 'dist',
	batch_size: int = 256,
):
	"""
	Calculates the importance scores for a specific token by permuting its embeddings
	and evaluating the impact on the model's performance.

	Performs permutation importance calculation for a specific token.

	Parameters:
	----------
	estimator : object
	    The machine learning model to evaluate.
	gnn : PreTrainedGNN
	    The pretrained GNN model used for featurization.
	X : list[pyg.data.Data]
	    List of graph data objects.
	y : array-like
	    Target values.
	token_idx : int
	    Index of the token to evaluate.
	n_repeats : int
	    Number of permutations to perform for the token.
	baseline_score : float or dict
	    Baseline score of the model without permutation.
	random_state : int
	    Random seed for reproducibility.
	scorer : str or callable
	    Scoring function or metric.
	perm_type : str
	    Type of permutation to apply ("dist" or "single").
	    The "single" type permutes at max one token at a time per graph,
	    while "dist" permutes across all tokens by distributing the random indices over the graph.
	    The default is "dist".

	Returns:
	-------
	np.ndarray or dict
	    Importance scores for the token.
	"""

	try:
		if check_token_presence(X, token_idx):
			dim = X[0].x.size(-1)
			random_indexes = get_permutations(
				dim=dim, repeats=n_repeats, random_state=random_state
			)
			scores = []
			for idx in random_indexes:
				X_permuted = [G.clone() for G in X]
				X_permuted = [
					permute_graph(G, token_idx, idx, perm_type=perm_type)
					for G in X_permuted
				]
				X_permuted = batch_graphs(X_permuted, batch_size=batch_size)
				if isinstance(X_permuted, list):
					embeddings = []
					for batch in X_permuted:
						batch = batch.to(gnn.device)
						embeddings.append(gnn(batch))
					embeddings = np.vstack(embeddings)
				else:
					X_permuted = X_permuted.to(gnn.device)
					embeddings = gnn(X_permuted)

				scores.append(
					_weights_scorer(
						scorer,
						estimator,
						embeddings,
						y,
						sample_weight=None,
					)
				)
				del X_permuted, embeddings
				torch.cuda.empty_cache()

			if isinstance(scores[0], dict):
				scores = _aggregate_score_dicts(scores)
			else:
				scores = np.array(scores)

		else:
			# put baseline scores when token not used
			if isinstance(scorer, _MultimetricScorer):
				scores = {
					sco: np.full(n_repeats, baseline_score[sco])
					for sco in scorer._scorers
				}
			else:
				scores = np.full((n_repeats), baseline_score)

		return scores
	except Exception as e:
		msg = (
			f'Failed at {token_idx}. \nError: {e}\nTraceback: {traceback.format_exc()}'
		)
		raise Exception(msg) from e


def token_importance(
	estimator,
	gnn: PreTrainedGNN,
	X: list[pyg.data.Data],
	y: np.ndarray,
	n_repeats: int = 5,
	random_state: int = 42,
	scorer: str | Callable = None,
	n_jobs=-1,
	sample_weight=None,
	perm_type='dist',
	batch_size: int = 256,
):
	"""
	Computes the importance of tokens by permuting their embeddings and evaluating the impact
	on the model's performance.

	Based on `sklearn.inspection.permutation_importance`.


	Parameters:
	----------
	estimator : object
	    The machine learning model to evaluate.
	gnn : PreTrainedGNN
	    The pretrained GNN model used for featurization.
	X : list[pyg.data.Data]
	    List of graph data objects.
	y : array-like
	    Target values.
	n_repeats : int, optional
	    Number of permutations to perform for each token (default is 5).
	random_state : int, optional
	    Random seed for reproducibility (default is 42).
	scorer : str or Callable, optional
	    Scoring function or metric (default is None).
	    E.g. "accuracy", "f1", or a custom scoring function.
	n_jobs : int, optional
	    Number of parallel jobs to run (default is -1).
	sample_weight : array-like, optional
	    Sample weights (default is None).
	perm_type : str, optional
	    Type of permutation to apply ("dist" or "single", default is "dist").

	Returns:
	-------
	dict or _create_importances_bunch
	    Token importance scores.
	"""
	num_tokens = len(gnn.transform.featurizer.node_types)
	scorer = check_scoring(estimator, scorer)
	baseline_X = gnn(X)
	[G.to('cpu') for G in X]
	torch.cuda.empty_cache()
	baseline_score = _weights_scorer(scorer, estimator, baseline_X, y, sample_weight)

	scores = Parallel(n_jobs=n_jobs, prefer='threads')(
		delayed(calculate_token_scores)(
			estimator=estimator,
			gnn=gnn,
			X=X,
			y=y,
			token_idx=token_idx,
			baseline_score=baseline_score,
			n_repeats=n_repeats,
			random_state=random_state,
			scorer=scorer,
			perm_type=perm_type,
			batch_size=batch_size,
		)
		for token_idx in range(num_tokens)
	)

	if isinstance(baseline_score, dict):
		return {
			name: _create_importances_bunch(
				baseline_score[name],
				np.array([scores[col_idx][name] for col_idx in range(num_tokens)]),
			)
			for name in baseline_score
		}
	torch.cuda.empty_cache()
	return _create_importances_bunch(baseline_score, np.array(scores))
