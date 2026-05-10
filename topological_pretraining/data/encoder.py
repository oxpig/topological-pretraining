from typing import Any

import numpy as np
from sklearn.base import BaseEstimator


class OneHotEncoder(BaseEstimator):
	"""
	One-hot encoder for categorical variables.

	Parameters
	----------
	categories: list
	    The categories to encode.

	Attributes
	----------
	categories_: list
	    The categories used for encoding.
	"""

	def __init__(self, categories: list | None = None):
		super().__init__()
		self.encoder = {}
		self.decoder = {}
		if categories is not None:
			self.fit(categories)

	@property
	def categories(self):
		"""
		Get the categories used for encoding.

		Returns
		-------
		out: list
		    The categories used for encoding.
		"""
		return list(self.encoder.keys())

	def fit(self, X: list):
		X = set(X)
		self.encoder, self.decoder = {}, {}
		for i, j in enumerate(X):
			self.encoder[j] = i
			self.decoder[i] = j
		self.encoder['UNK'] = len(X)
		self.decoder[len(X)] = 'UNK'

	def transform(self, X: list[Any] | Any):
		if not isinstance(X, list):
			X = [X]
		out = np.zeros((len(X), len(self.encoder)), dtype=int)
		for i, j in enumerate(X):
			out[i, self.encoder.get(j, self.encoder['UNK'])] = 1
		return out

	def inverse_transform(self, X):
		out = []
		for i in X:
			out.append(self.decoder[np.argmax(i)])
		return out
