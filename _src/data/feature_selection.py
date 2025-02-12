import numpy as np

class CoCorr:
    """
    Fetaure selection based on collinearity.

    For highly correlated features, the feature with the lowest variance is removed.

    Parameters
    ----------
    threshold : float
        The threshold for collinearity. Default is 0.9.

    Attributes
    ----------
    threshold : float
        The threshold for collinearity.
    to_keep : list
        The indices of the features to keep.
    """
    def __init__(self, threshold: float = 0.9):
        self.threshold = threshold
        self.to_keep = []
        super().__init__()

    def fit(self, X: np.ndarray):
        """
        Determine which features to keep based on multilinearity and variance.

        Parameters
        ----------
        X : np.ndarray
            The input data. The shape is (n_samples, n_features).
        """
        var = X.std(axis=0)
        corr_matrix = np.corrcoef(X, rowvar=False)
        upper = np.triu(corr_matrix, k=1)
        idx = np.where(upper >= self.threshold)
        idx = zip(*idx)
        exclude = []
        for i, j in idx:
            out = i if var[i] < var[j] else j
            exclude.append(int(out))
        self.to_keep = np.array([i for i in range(X.shape[1]) if i not in set(exclude)])

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the input data by removing highly correlated features.

        Parameters
        ----------
        X : np.ndarray
            The input data. The shape is (n_samples, n_features).

        Returns
        -------
        np.ndarray
            The transformed data. The shape is (n_samples, len(self.to_keep)).
        """
        return X[:, self.to_keep]

    def fit_transform(self, X: np.ndarray):
        """
        Fit CoCorr to X and return the transformed X.

        Parameters
        ----------
        X : np.ndarray
            The input data. The shape is (n_samples, n_features).

        Returns
        -------
        np.ndarray
            The transformed data. The shape is (n_samples, len(self.to_keep)).
        """
        self.fit(X)
        return self.transform(X)

    
class SelectAll:
    """
    Dummy class to select all features.
    """
    named_steps = {}
    def fit(self, X: np.ndarray, y: np.ndarray = None):
        pass

    def transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        return X if y is None else (X, y)

    def fit_transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        return X if y is None else (X, y)