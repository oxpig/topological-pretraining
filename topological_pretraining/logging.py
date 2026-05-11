import numpy as np


class Logger(dict):
	"""
	A simple logging class that extends a dictionary for saving and loading hyperparameters and scores.

	This project originally used Neptune for logging experiments, however Neptune has since been discontinued and is no longer available to use.
	This Logger class is a basic replacement for Neptune so that loss and scores from training and benchmarking can still be saved and accessed.

	Parameters:
	----------
	path : Optional[str]
	    The file path to save the log. If None, the log will not be saved to disk. Supported formats are '.npy' and '.npz'.
	args : Any
	    Additional arguments for the dictionary.
	kwargs : Any
	    Additional keyword arguments for the dictionary.

	Methods:
	-------
	save()
	    Saves the log to disk if a path is provided.
	load()
	    Loads the log from disk if a path is provided.
	"""

	def __init__(self, path: str | None = None, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.path = path

	def save(self):
		if self.path is not None:
			if self.path.suffix == '.npy':
				np.save(self.path, self)
			elif self.path.suffix == '.npz':
				np.savez_compressed(self.path, **self)
			else:
				raise ValueError(
					"Unsupported file format. Please use '.npy' or '.npz'."
				)

	def load(self):
		if self.path is not None:
			if self.path.suffix == '.npy':
				loaded = np.load(self.path, allow_pickle=True).item()
			elif self.path.suffix == '.npz':
				loaded = np.load(self.path, allow_pickle=True)
				loaded = {key: loaded[key] for key in loaded.files}
			else:
				raise ValueError(
					"Unsupported file format. Please use '.npy' or '.npz'."
				)
			self.update(loaded)
