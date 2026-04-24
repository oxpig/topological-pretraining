from lightgbm import LGBMClassifier, LGBMRegressor, LGBMModel
from sklearn.base import BaseEstimator


class LGBM(BaseEstimator):
    """
    LightGBM model wrapper for classification and regression tasks.
    
    Parameters:
    ----------
    task : Literal['classification', 'regression']
        The type of task to perform. Can be 'classification' or 'regression'.
    seed : int, optional
        Random seed for reproducibility. Default is 42.
    logging : dict, optional
        For compatibility with sklearn_gin logging.
    device : str, optional
        Device to use for training. Can be 'cpu' or 'gpu'. Default is 'cpu'.
    proba_as_pred : bool, optional
        If True, returns probabilities for classification tasks. Default is True.
    **kwargs : Any
        Additional keyword arguments for the LightGBM model.
    """
    def __init__(
        self, task, seed=42,
        logging=None,
        device='cpu',
        proba_as_pred=True,
        **kwargs
    ):
        if task == 'classification':
            if 'is_unbalance' not in kwargs:
                kwargs['is_unbalance'] = True
            self.model = LGBMClassifier(random_state=seed, **kwargs)
        elif task == 'regression':
            self.model = LGBMRegressor(random_state=seed, **kwargs)
        
        self.task = task
        self.logging = None
        self.proba_as_pred = proba_as_pred
        if device == 'cuda':
            device = 'gpu'
        self.kwargs = kwargs
        self.kwargs['device'] = device

    
    def transform(self, X):
        """
        Transform the input data using the LightGBM model.

        Parameters:
        ----------
        X : np.ndarray
            Input data to transform.
        Returns:
        -------
        np.ndarray
            Transformed data.
        """
        return self.model.transform(X)

    def fit(self, X, y):
        """
        Fit the LightGBM model to the input data.

        Parameters:
        ----------
        X : np.ndarray
            Input data to fit the model.
        y : np.ndarray
            Target values for the input data.

        Returns:
        -------
        None
            The model is fitted in place.
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predict using the fitted LightGBM model.

        Parameters:
        ----------
        X : np.ndarray
            Input data for prediction.

        Returns:
        -------
        np.ndarray
            Predicted values. If task is 'classification' and proba_as_pred is True,
            returns probabilities; otherwise, returns class labels or regression values.
        """
        if self.task == 'classification' and self.proba_as_pred:
            # to make sure outputs are the same format as regression
            # want to store predictions as probabilities; more versatile when looking at metrics
            return self.model.predict_proba(X)[:, 1]
        
        else:
            return self.model.predict(X)
        
    def predict_proba(self, X):
        """
        Predict probabilities using the fitted LightGBM model.

        Parameters:
        ----------
        X : np.ndarray
            Input data for probability prediction.

        Returns:
        -------
        np.ndarray
            Predicted probabilities for each class.

        Raises:
        -------
        ValueError
            If the task is not 'classification'.
        """
        if self.task == 'classification':
            return self.model.predict_proba(X)
        
        else:
            raise ValueError("predict_proba is only available for classification tasks.")

    def predict_class(self, X):
        """
        Predict class labels using the fitted LightGBM model.

        Parameters:
        ----------
        X : np.ndarray
            Input data for class prediction.

        Returns:
        -------
        np.ndarray
            Predicted class labels.

        Raises:
        -------
        ValueError
            If the task is not 'classification'.
        """
        if self.task == 'classification':
            return self.model.predict(X)
        
        else:
            raise ValueError("class_predict is only available for classification tasks.")
        
    def get_feature_importance(self):
        """
        Get GINI feature importances from the fitted LightGBM model.

        Returns:
        -------
        np.ndarray
            Feature importances as a numpy array.
        """
        return self.model.feature_importances_

    @property
    def feature_importances(self):
        """
        Property to access feature importances.
        """
        return self.get_feature_importance()

    def __sklearn_tags__(self):
        """
        Get the sklearn tags for the LightGBM model.

        Returns:
        -------
        dict
            A dictionary containing the sklearn tags for the model.
        """
        return self.model.__sklearn_tags__()
    
    def __getattr__(self, attr):
        """
        Get attributes from the underlying LightGBM model.

        Parameters:
        ----------
        attr : str
            The attribute name to retrieve from the LightGBM model.
        """
        return self.model.__getattribute__(attr)

        
    def __repr__(self):
        """
        Returns a string representation of the LGBM model.
        """
        return f"LGBM(task={self.task}, kwargs={self.kwargs})"
    
