from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.base import BaseEstimator

class LGBM:
    def __init__(self, task, **kwargs):
        self.task = task
        if task == 'classification':
            self.model = LGBMClassifier(**kwargs)
        elif task == 'regression':
            self.model = LGBMRegressor(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    