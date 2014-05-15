from __future__ import division
 
__author__ = 'Miroslaw Horbal'
__email__ = 'miroslaw@gmail.com'
__date__ = '09-08-2013'
 
import numpy as np
from scipy import optimize, sparse
 
OPTIMIZATION_FUNCTIONS = { 'cg':   optimize.fmin_cg,
                           'bfgs': optimize.fmin_bfgs }
 
class BaseModel(object):
    parameters_ = []
   
    def __repr__(self):
        strs = ['%s=%s' % (param, str(getattr(self, param)))
                 for param in self.parameters_]
        return self.__class__.__name__ + '(' + ', '.join(strs) + ')'
       
    def fit(self, X, y=None, coef=None):
        raise NotImplementedError('%s.fit not implemented' % self.__class.__name__)
   
    def predict(self, X):
        raise NotImplementedError('%s.predict not implemented' % self.__class.__name__)
   
    def score(self, X, y):
        raise NotImplementedError('%s.score not implemented' % self.__class.__name__)
       
class BaseDescentModel(BaseModel):
    parameters_ = ['opt', 'maxiter', 'tol', 'verbose']
   
    def __init__(self, opt='cg', maxiter=1000, tol=1e-4, verbose=False,
                  callback_data=None):
        """
       Parameters:
         opt - optimization algorithm to use for gardient decent
               options are 'cg', 'bfgs' (default 'bfgs')
         maxiter - maximum number of iterations (default 1000)
         tol - terminate optimization if gradient l2 is smaller than tol (default 1e-4)
         verbose - display convergence information at each iteration (default False)
       """
        self.opt = opt
        self.maxiter = maxiter
        self.tol = tol
        self.verbose = verbose
        self.callback_data = callback_data
       
    @property
    def opt(self):
        """Optimization algorithm to use for gradient decent"""
        return self._opt
   
    @opt.setter
    def opt(self, o):
        """
       Set the optimization algorithm for gradient decent
       
       Parameters:
         o - 'cg' for conjugate gradient decent
             'bfgs' for BFGS algorithm
       """
        if o not in OPTIMIZATION_FUNCTIONS:
            raise Error('Unknown optimization routine %s' % o)
        self._opt = o
        self._optimize = OPTIMIZATION_FUNCTIONS[o]
   
    def _callback(self, X, y):
        """
       Helper method that generates a callback function for the optimization
       algorithm opt if verbose is set to True
       """
        def callback(coef):
            self.i += 1
            self.coef_ = np.reshape(coef, self._coef_shape)
            score = self.score(X, y)
            print 'iter %i | Score: %f\r' % (self.i, score)
        self.i = 0
        return callback if self.verbose else None
