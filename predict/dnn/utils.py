import numpy as np
import theano as th
import theano.tensor as tht


def sigmoid(z):
    return tht.nnet.sigmoid(z)

def tanh(z):
    return tht.nnet.tanh(z)

def relu(z):
    return tht.maximum(0, z)

def nll(Y, Z):
    nll = -tht.sum(Y * tht.log(Z) + (1 - Y) * tht.log(1 - Z))
    return nll

def wnll(Y, Z, wc, ws):
    t = Y
    nll = -tht.sum(ws.dot(wc[1] * Y * tht.log(Z) +\
                          wc[0] * (1 - Y) * tht.log(1 - Z)))
    return nll

def norm_center(X):
    return X - X.mean(axis=0)

def norm_scale(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    Xn = (X - mu) / sd
    return (Xn, mu, sd)

def norm_minmax(X, min_=0, max_=1):
    Xmin = X.min(axis=0)
    Xmax = X.max(axis=0)
    return (X - Xmin) / (Xmax - Xmin) * (max_ - min_) + min_



class UpdatesGenerator(object):

    def __init__(self, params, dparams):
        self.params = params
        self.dparams = dparams

    def sgd(self, lr=0.01):
        updates = []
        for param, dparam in zip(self.params, self.dparams):
            updates.append([param, param - lr * dparam])
        return updates

    def momentum(self, lr=0.01, mom=0.9):
        updates = []
        for param, dparam in zip(self.params, self.dparams):
            step = th.shared(np.zeros(param.get_value().shape), borrow=True)
            step_new = mom * step - (1 - mom) * lr * dparam
            updates.append([param, param + step_new])
            updates.append([step, step_new])
        return updates

    def adagrad(self, lr=0.01, mom=0.9, eps=1e-6):
        updates = []
        for param, dparam in zip(self.params, self.dparams):
            acc = th.shared(np.zeros(param.get_value().shape), borrow=True)
            a = mom * acc + (1 - mom) * dparam**2
            step = - lr * dparam / tht.sqrt(a + eps)
            updates.append([param, param + step])
            updates.append([acc, a])
        return updates

    def adadelta(self, mom=0.95, eps=1e-6):
        updates = []
        for param, dparam in zip(self.params, self.dparams):
            t = param.get_value().shape
            acc_grad = th.shared(np.zeros(t), borrow=True)
            acc_step = th.shared(np.zeros(t), borrow=True)
            acc_grad_new = mom * acc_grad + (1 - mom) * dparam**2
            step = - tht.sqrt((acc_step + eps) / tht.sqrt(acc_grad_new + eps)) * dparam
            acc_step_new = mom * acc_step + (1 - mom) * step**2
            updates.append([param, param + step])
            updates.append([acc_grad, acc_grad_new])
            updates.append([acc_step, acc_step_new])
        return updates


class WeightsInitializer(object):

    def __init__(self, nin, nout, rng=0):
        self.nin = nin
        self.nout = nout
        self.shape = (nin, nout)
        if isinstance(rng, int):
            rng = np.random.RandomState(rng)
        self.rng = rng

    def normal(self, mean=0.0, std=1.0):
        w = self.rng.normal(self.mean, self.std, self.shape)
        return w

    def uniform(self, scale=1.0):
        c = np.sqrt(6.0 / (self.nin + self.nout))
        if scale:
            c *= scale
        w = self.rng.uniform(-c, c, self.shape)
        return w

    def sigmoid(self):
        return self.uniform(4.0)

    def tanh(self):
        return self.uniform()

    def relu(self):
        return self.uniform()


class Sgd(object):

    def __init__(self, batch_size=100, max_epochs=100, shuffle=True, rng=0):
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        if isinstance(rng, int):
            rng = np.random.RandomState(rng)
        self.shuffle = shuffle
        self.rng = rng

    def optimize(self, nsamples, train_fun, epoch_fun=None):
        nbatches = int(np.ceil(nsamples / self.batch_size))
        for epoch in range(self.max_epochs):
            shuffle = np.arange(nsamples)
            if self.shuffle:
                self.rng.shuffle(shuffle)
            for batch in range(nbatches):
                s = batch * self.batch_size
                e = s + self.batch_size
                train_fun(shuffle[s:e])
            if epoch_fun is not None:
                if epoch_fun(epoch):
                    break
        return epoch
