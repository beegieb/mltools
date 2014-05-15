from core.generalized import GeneralizedModel
from core.trainer import Trainer
from utils.functions import *
from utils.helpers import initialize_weights

from scipy import *
from numpy.random import normal, permutation, rand, uniform

LAYER_MODEL_FNS = { 'binary': sigmoid, 
                    'linear': linear }

LAYER_SAMPLE_FNS = { 'binary': sample_bernoulli, 
                     'linear': linear }

class RBM(GeneralizedModel):
    attrs_ = ['trainfn', 'n', 'batch_size', 'epochs', 'learn_rate', 'beta', 
              'momentum', 'verbose', 'hidden_size', 'visible_size', 
              'hidden_layer', 'visible_layer', 'dropout']
    
    def __init__(self, visible_size, hidden_size, epochs=1, learn_rate=0.1, 
                 trainfn='cdn', n=1, beta=0.0001, momentum=0., batch_size=10, 
                 visible_layer='binary', hidden_layer='binary', dropout=0.0, 
                 verbose=0):
        # Initialize args
        self.trainfn = trainfn
        self.epochs = epochs
        self.n = n
        self.learn_rate = learn_rate
        self.beta = beta
        self.batch_size = batch_size
        self.momentum = momentum
        self.verbose = verbose
        self.visible_size = visible_size
        self.hidden_size = hidden_size
        self.visible_layer = visible_layer
        self.hidden_layer = hidden_layer
        self.dropout = dropout
        # Initialize Biases and Weights
        self.vbias = zeros(visible_size)
        self.hbias = zeros(hidden_size)
        self.W = initialize_weights(visible_size, hidden_size)
        self.prevgrad = {'W': zeros(self.W.shape), 
                         'hbias': zeros(hidden_size), 
                         'vbias': zeros(visible_size)}
        self.p = np.zeros((self.batch_size, self.hidden_size))
        if self.trainfn == 'fpcd':
            self.fW = zeros(self.W.shape)
            self.flr = self.learn_rate*exp(1) #fast learn rate heuristic
            self.fWd = 49./50 #fast weight decay heuristic
        # Initialize Trainer instance
        self.trainer = Trainer()
        
    def params(self):
        return {'W': self.W, 'hbias': self.hbias, 'vbias': self.vbias}
    
    def propup(self, vis, fw=False):
        f = LAYER_MODEL_FNS[self.hidden_layer]
        g = LAYER_SAMPLE_FNS[self.hidden_layer]
        W = self.fW + self.W if fw else self.W
        pre_non_lin = vis.dot(W.T) + self.hbias
        non_lin = f(pre_non_lin)
        if self.dropout > 0.0:
            activs = uniform(0, 1, size=non_lin.shape) >= self.dropout
            non_lin *= activs
        sample = g(non_lin) if self.hidden_layer != 'NRLU' else g(pre_non_lin * activs)
        return (sample, non_lin, pre_non_lin)
    
    def propdown(self, hid, fw=False):
        f = LAYER_MODEL_FNS[self.visible_layer]
        g = LAYER_SAMPLE_FNS[self.visible_layer]
        W = self.fW + self.W if fw else self.W
        pre_non_lin = hid.dot(W) + self.vbias
        non_lin = f(pre_non_lin)
        sample = g(non_lin)
        return (sample, non_lin, pre_non_lin)

    def gibbs_hvh(self, h, mf=False, **args):
        v_samples = self.propdown(h, **args)
        v = v_samples[1] if mf else v_samples[0]
        h_samples = self.propup(v, **args)
        return v_samples, h_samples
    
    def gibbs_vhv(self, v, mf=False, **args):
        h_samples = self.propup(v, **args)
        h = h_samples[1] if mf else h_samples[-1]
        v_samples = self.propdown(h, **args)
        return v_samples, h_samples
    
    def cost(self, v):
        if len(np.shape(v)) == 1: v.shape = (1,len(v))
        use_fw = self.trainfn == 'fpcd'
        use_persist = use_fw or self.trainfn == 'pcd'
        num_points = v.shape[0]
        # positive phase
        pos_h_samples = self.propup(v)
        # negative phase
        nh0 = self.p[:num_points] if use_persist else pos_h_samples[0]
        for i in range(self.n):
            neg_v_samples, neg_h_samples = self.gibbs_hvh(nh0, fw=use_fw)
            nh0 = neg_h_samples[0]
        # compute gradients
        grad = self.grad(v, pos_h_samples, neg_v_samples, neg_h_samples)
        self.p[:num_points] = nh0
        # compute reconstruction error
        if self.trainfn=='cdn':
            reconstruction = neg_v_samples[1]
        else:
            reconstruction = self.propdown(pos_h_samples[0])[1]
        cost = np.sum(np.square(v - reconstruction)) / self.batch_size
        return cost, grad
        
    def update(self, grad):
        prev_grad = self.prevgrad
        dW = self.momentum * prev_grad['W'] + \
             self.learn_rate * (grad['W'] - self.beta * self.W)
        dh = self.momentum * prev_grad['hbias'] + \
             self.learn_rate * grad['hbias']
        dv = self.momentum * prev_grad['vbias'] + \
             self.learn_rate * grad['vbias']
        self.W += dW
        self.hbias += dh
        self.vbias += dv
        # Fast weight update for PCD
        if self.trainfn == 'fpcd':
            self.fW = self.fWd * self.fW + self.flr * grad['W'] 
        self.prevgrad['W'] = dW
        self.prevgrad['hbias'] = dh
        self.prevgrad['vbias'] = dv
        return self
    
    def grad(self, pv0, pos_h, neg_v, neg_h):
        grad = {}
        num_points = pv0.shape[0]
        E_v = neg_v[1]
        E_h = neg_h[1]
        E_hgv = pos_h[1]
        E_vh = np.dot(E_h.T, E_v)
        E_vhgv = np.dot(E_hgv.T, pv0)
        grad['W'] = (E_vhgv - E_vh) / num_points
        grad['vbias'] = mean(pv0 - E_v, 0)
        grad['hbias'] = mean(E_hgv - E_h, 0)
        return grad
    
    def E(self, v0, h0):
        if len(shape(v0)) == 1: v0.shape = (1,len(v0))
        if len(shape(h0[0])) == 1: h0.shape = (1,len(h0[0]))
        if self.visible_layer == 'linear':
            vis_e = sum(square(self.vbias - v0))/2
        else:
            vis_e = -sum(self.vbias * v0)
        if self.hidden_layer == 'linear':
            hid_e = sum(square(self.hbias - h0))/2
        else:
            hid_e = -sum(self.hbias * h0)
        vishid_e = -sum(dot(h0[0].T, v0) * self.W)
        return hid_e + vishid_e

    def F(self, v0):
        if len(shape(v0)) == 1: v0.shape = (1,len(v0))
        X = dot(v0, self.W.T) + self.hbias
        return -dot(v0, self.vbias) - sum(log(1 + exp(X)))

    def train(self, data, max_iter=1):
        args = { 'epochs': self.epochs,
                 'batch_size': self.batch_size,
                 'max_iter': max_iter,
                 'verbose': self.verbose }
        return self.trainer.train(self, data, **args)
