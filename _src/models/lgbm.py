from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.base import BaseEstimator

class LGBM:
    def __init__(self, task, seed=42, **kwargs):
        self.task = task
        if task == 'classification':
            if 'is_unbalance' not in kwargs:
                kwargs['is_unbalance'] = True
            self.model = LGBMClassifier(random_state=seed, **kwargs)
        elif task == 'regression':
            self.model = LGBMRegressor(random_state=seed, **kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        if self.task == 'classification':
            return self.model.predict_proba(X)[:, 1]
        
        else:
            return self.model.predict(X)
