from linear_models import LinearMAE
import numpy as np
from scipy import optimize, sparse

class NodeRegressor(object):
    def __init__(self, max_depth=1, min_sample=2, split=np.median,
                  model=LinearMAE, **model_args):
        self._model_type = model 
        self._model = model(**model_args)
        self._model_args = model_args
        self.lchild = None
        self.rchild = None
        self.max_depth = max_depth
        self.min_sample = min_sample 
        self.split = split
    
    def fit(self, X, y):
        self._model = self._model.fit(X, y)
        pred = self._model.predict(X)
        if self.max_depth > 1:
            self._construct_children()
            self.split_ = self.split(pred)
            if sum(pred < self.split_) >= self.min_sample:
                self.lchild.fit(X[find(pred < self.split_)], y[pred < self.split_])
            else:
                self.lchild = None
            if sum(pred >= self.split_) >= self.min_sample:
                self.rchild.fit(X[find(pred >= self.split_)], y[pred >= self.split_])
            else:
                self.rchild = None
                
        return self 
    
    def predict(self, X):
        pred = self._model.predict(X) 
        if self.max_depth > 1:
            new_pred = np.copy(pred)
            if self.lchild is not None:
                new_pred[pred < self.split_] = self.lchild.predict(X[find(pred < self.split_)])
            if self.rchild is not None:
                new_pred[pred >= self.split_] = self.rchild.predict(X[find(pred >= self.split_)])
            pred = new_pred 
        return pred 
        
    def _construct_children(self):
        self.lchild = NodeRegressor(max_depth=self.max_depth - 1,
                           min_sample=self.min_sample,
                           model=self._model_type,
                           **self._model_args)
        self.rchild = NodeRegressor(max_depth=self.max_depth - 1,
                           min_sample=self.min_sample,
                           model=self._model_type,
                           **self._model_args)
