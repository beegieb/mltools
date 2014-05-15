from __future__ import division
 
__author__ = 'Miroslaw Horbal'
__email__ = 'miroslaw@gmail.com'
__date__ = '09-08-2013'
 
import numpy as np
from scipy import optimize, sparse
from base import BaseDescentModel
 
OPTIMIZATION_FUNCTIONS = { 'cg':   optimize.fmin_cg,
                           'bfgs': optimize.fmin_bfgs }
 
def mae(h, y):
    return np.abs(h - y).mean()

def cost_mae(coef, X, y, l2=0):
    """
    Parameters:
      coef - the weights of the linear model must have size (n + 1)*o
      X - data array for the linear model. Has shape (m,n)
      y - output target array for the linear model. Has shape (m,o)
      l2 - magnitude of the l2 penalty
    """
    m, n = X.shape
    Xb = np.hstack((np.ones((m,1)), X))
    pred = Xb.dot(coef)
    cost = mae(pred, y)
    #c_l2 = np.mean(np.square(coef[1:])) if l2 > 0 else 0
    return cost #+ 0.5 * l2 * c_l2
   
def grad_mae(coef, X, y, l2=0):
    """
    Compute the gradient of a linear model with RMSLE:
   
    Parameters:
      coef - the weights of the linear model must have size (n + 1)*o
      X - data array for the linear model. Has shape (m,n)
      y - output target array for the linear model. Has shape (m,o)
      l2 - magnitude of the l2 penalty
    """
    m, n = X.shape
    Xb = np.hstack((np.ones((m,1)), X))
    pred = Xb.dot(coef)
    err = pred - y
    derr = np.sign(err)
    dcoef = (derr * Xb.T).mean(1)
    return dcoef.flatten()
    
class LinearMAE(BaseDescentModel):
    parameters_ = ['l2'] + BaseDescentModel.parameters_
    def __init__(self, l2=0.0, **args):
        """
       Parameters:
         l1 - magnitude of l1 penalty (default 0.0)
         l2 - magnitude of l2 penalty (default 0.0)
         args - arguments to pass to the optimization routine
                see BaseDescentModel help
       """
        BaseDescentModel.__init__(self, **args)
        self.l2 = l2
   
    def _make_coef(self, X, y, coef=None):
        m, n = X.shape
        if coef is None:
            coef = np.zeros((n+1,))
        elif coef.shape != (n+1,):
            raise Error('coef must be None or be shape %s' % (str((n+1,))))
        self._coef_shape = coef.shape
        return coef
       
    def score(self, X, y):
        """
       Compute the RMSLE of the linear model prediction on X against y
       
       Must only be run after calling fit
       
       Parameters:
         X - data array for the linear model. Has shape (m,n)
         y - output target array for the linear model. Has shape (m,o)
       """
        pred = self.predict(X)
        return mae(pred, y)
   
    def predict(self, X):
        """
        Compute the linear model prediction on X
       
        Must only be run after calling fit
       
        Parameters:
          X - data array for the linear model. Has shape (m,n)
        """
        m, n = X.shape
        Xb = np.hstack((np.ones((m,1)), X))
        coef = self.coef_
        pred = Xb.dot(coef)
        return pred
   
    def fit(self, X, y, coef=None):
        """
       Fit the linear model using gradient decent methods
       
       Parameters:
         X - data array for the linear model. Has shape (m,n)
         y - output target array for the linear model. Has shape (m,o)
         coef - None or array of size (n+1) * o
       
       Sets attributes:
         coef_ - the weights of the linear model
       """
        coef = self._make_coef(X, y, coef)
        cX, cy = (X, y) if self.callback_data is None else self.callback_data
        coef = self._optimize(f=cost_mae,
                              x0=coef.flatten(),
                              fprime=grad_mae,
                              args=(X, y, self.l2),
                              gtol=self.tol,
                              maxiter=self.maxiter,
                              disp=0,
                              callback=self._callback(cX,cy))
        self.coef_ = np.reshape(coef, self._coef_shape)
        return self
