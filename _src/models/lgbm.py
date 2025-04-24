from lightgbm import LGBMClassifier, LGBMRegressor

class LGBM:
    def __init__(
        self, task, seed=42,
        neptune_run=None, neptune_location=None,
        device='cpu',
        **kwargs
    ):
        self.task = task
        self.neptune_run = neptune_run # for compatibility with other models
        self.neptune_location = neptune_location # for compatibility with other models
        if device == 'cuda':
            device = 'gpu'
        self.kwargs = kwargs
        self.kwargs['device'] = device
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
            # to make sure outputs are the same format as regression
            # want to store pr
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