from lightgbm import LGBMClassifier, LGBMRegressor

class LGBM:
    def __init__(
        self, task, seed=42,
        neptune_run=None, neptune_location=None,
        **kwargs
    ):
        self.task = task
        self.neptune_run = neptune_run # for compatibility with other models
        self.neptune_location = neptune_location # for compatibility with other models
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
