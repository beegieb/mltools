from scipy import exp, sum, sqrt, pi
import scipy as np

def sigmoid(X):
    """
    numpy.array -> numpy.array
    
    Compute sigmoid function: 1 / (1 + exp(-X))
    """
    return 1 / (1 + exp(-X))

def sigmoid_grad(X):
    sig = sigmoid(X)
    return sig * (1-sig)

def tanh_grad(X):
    return 1 / np.square(np.cosh(X))

def linear_grad(X):
    return np.ones(X.shape)

def rectified_linear_grad(X):
    return (X > 0).astype('b')
    
def softmax(X):
    """
    numpy.array -> numpy.array
    
    Compute softmax function: exp(X) / sum(exp(X))
    """
    mx = X.max()
    ex = exp(X.T - mx)
    return (ex / sum(ex,0)).T

def linear(X):
    return X

def sample_bernoulli(X):
    """
    numpy.array -> numpy.array
    
    All values of X must be probabilities of independent events occuring according
    to the binomial distribution
    
    Returns an indicator array of the same shape as input recording with
    output[i,j] = 1 iif X[i,j] >= uniform(0, 1)
    """
    return (X >= np.random.uniform(size=X.shape)).astype('b')

def cross_entropy(X, Y):
    return -np.sum((1 - Y) * np.log(1-X) + Y * np.log(X))

def square_error(X, Y):
    return np.sum(np.square(X - Y))
