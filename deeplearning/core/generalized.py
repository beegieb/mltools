import pickle
import scipy as np

from trainer import Trainer

class GeneralizedModel(object):
    """
    Base class for learning models. All model types inheriting this object must
    include:
        : attrs_ class variable as an iterable of attributes to display for
          pretty printing (probably a better way to do this)
        : cost function that takes the model and data (and possibly labels)
          and computes the cost of the data given the model. Must return
          a numeric cost and a gradient
        : train function that takes the model and data (and possibly labels)
          and transforms the data into a suitable format to pass into an 
          optimization algorithm
        : update function that takes the model and gradient to update the model
          parameters
    
    This class adds support for pretty printing and saving using pythons
    pickle interface
    """
    attrs_ = []
    
    def __repr__(self):
        L = []
        for attr in self.attrs_:
            L.append(attr + '=' + str(getattr(self, attr)))
        return self.__class__.__name__ + '(' + ', '.join(L) + ')'

    def cost(self, X):
        raise NotImplementedError('cost not implemented')
    
    def train(self, X):
        raise NotImplementedError('train not implemented')
    
    def update(self, grad):
        raise NotImplementedError('update not implemented')
        
    def save(self, filename, mode=2):
        f = open(filename, 'w')
        classname = self.__class__.__name__
        print 'Saving %s to %s under mode %i' % (self.__class__.__name__, 
                                                 filename, mode)
        pickle.dump(self, f, mode)
        f.close()
