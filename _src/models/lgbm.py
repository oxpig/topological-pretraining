from lightgbm import LGBMClassifier, LGBMRegressor, LGBMModel
from sklearn.base import BaseEstimator


class LGBM(BaseEstimator):
    def __init__(
        self, task, seed=42,
        neptune_run=None, neptune_location=None,
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
        self.neptune_run = neptune_run # for compatibility with other models
        self.neptune_location = neptune_location # for compatibility with other models
        self.proba_as_pred = proba_as_pred
        if device == 'cuda':
            device = 'gpu'
        self.kwargs = kwargs
        self.kwargs['device'] = device

    
    def transform(self, X):
        return self.model.transform(X)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        if self.task == 'classification' and self.proba_as_pred:
            # to make sure outputs are the same format as regression
            # want to store predictions as probabilities; more versatile when looking at metrics
            return self.model.predict_proba(X)[:, 1]
        
        else:
            return self.model.predict(X)
        
    def predict_proba(self, X):
        if self.task == 'classification':
            return self.model.predict_proba(X)
        
        else:
            raise ValueError("predict_proba is only available for classification tasks.")

    def predict_class(self, X):
        if self.task == 'classification':
            return self.model.predict(X)
        
        else:
            raise ValueError("class_predict is only available for classification tasks.")
        
    def get_feature_importance(self):
        return self.model.feature_importances_

    @property
    def feature_importances(self):
        return self.get_feature_importance()

    def __sklearn_tags__(self):
        return self.model.__sklearn_tags__()
    
    def __getattr__(self, attr):
        return self.model.__getattribute__(attr)

        
    def __repr__(self):
        return f"LGBM(task={self.task}, kwargs={self.kwargs})"
    
