import os.path as pt
import numpy as np
import pandas as pd
import sklearn.metrics as skm
from sklearn.grid_search import ParameterGrid
import copy

__dir = pt.dirname(pt.realpath(__file__))
# sys.path.insert(0, pt.join(__dir, '../module'))


def cor(y, z):
    return np.corrcoef(y, z)[0, 1]

def mse(y, z):
    return np.mean((y - z)**2)

def rmse(y, z):
    return np.sqrt(mse(y, z))

def auc(y, z):
    return skm.roc_auc_score(y, z)

def acc(y, z, r=True):
    if r:
        z = np.round(z)
    return skm.accuracy_score(y, z)

def tpr(y, z, r=True):
    if r:
        z = np.round(z)
    return skm.recall_score(y, z)

def tnr(y, z, r=True):
    if r:
        z = np.round(z)
    c = skm.confusion_matrix(y, z)
    return c[0, 0] / c[0].sum()

def mcc(y, z, r=True):
    if r:
        z = np.round(z)
    return skm.matthews_corrcoef(y, z)


def complete_cases(x, y=None):
    x = [x]
    if y is not None:
        x.append(y)
    x = [np.asarray(x_) for x_ in x]
    h = None
    for x_ in x:
        if len(x_.shape) == 1:
            hx = ~np.isnan(x_)
        else:
            hx = ~np.any(np.isnan(x_), axis=1)
        if h is None:
            h = hx
        else:
            h &= hx
    xc = [x_[h] for x_ in x]
    return xc

def score(Y, Z, fun=skm.roc_auc_score):
    y = np.asarray(Y).ravel()
    z = np.asarray(Z).ravel()
    y, z = complete_cases(y, z)
    return fun(y, z)

def scores(Y, Z, fun=skm.roc_auc_score):
    Y = np.asarray(Y)
    Z = np.asarray(Z)
    assert Y.shape == Z.shape
    num_tasks = Y.shape[1]
    s = []
    for task in range(num_tasks):
        y = Y[:, task]
        z = Z[:, task]
        y, z = complete_cases(y, z)
        s.append(fun(y, z))
    return s


def scores_frame(Y, Z,
                 funs=[('auc', auc),
                       ('acc', acc),
                       ('tpr', tpr),
                       ('tnr', tnr),
                       ('mcc', mcc),
                       ('rmse', rmse),
                       ('cor', cor)]
                 ):
    s = dict()
    l = []
    for k, v in funs:
        s[k] = scores(Y, Z, v)
        l.append(k)
    s = pd.DataFrame(s, index=Y.columns, columns=l)
    s.index.name = 'sample'
    s.reset_index(inplace=True)
    return s


def scores_frame_reg(Y, Z,
                     funs={'mse': mse, 'cor': cor}, order=['mse', 'cor']):
    return scores_frame(Y, Z, funs=funs, order=order)


def flatten_index(index, sep='_'):
    if index.nlevels > 1:
        rv = [sep.join(x) for x in index.values]
    else:
        rv = index.values
    return rv



def holdout_opt(model, param_grid, train_X, train_Y, val_X, val_Y, fun=skm.roc_auc_score):
    param_names = list(param_grid.keys())
    param_grid = ParameterGrid(param_grid)
    opt_model = None
    max_score = None
    scores = dict()
    for x in param_names + ['train', 'val']:
        scores[x] = []
    for params in param_grid:
        model.set_params(**params)
        model.fit(train_X, train_Y)
        if fun in [mse, cor]:
            pf = model.predict
        else:
            pf = model.predict_proba
        def score_fun(X, Y):
            return score(Y, pf(X), fun)
        s = []
        s.append(score_fun(train_X, train_Y))
        s.append(score_fun(val_X, val_Y))
        if max_score is None or s[1] > max_score:
            opt_model = copy.deepcopy(model)
            max_score = s[1]
        for k, v in params.items():
            scores[k].append(v)
        scores['train'].append(s[0])
        scores['val'].append(s[1])
    scores = pd.DataFrame(scores)
    return (opt_model, scores)



class MultitaskClassifier(object):

    def __init__(self, m):
        self.model = m

    def X_(self, X, task):
        return X[:, self.task_features[task]]

    def Xy_(self, X, Y, task):
        y = Y[:, task]
        h = ~np.isnan(y)
        y = y[h]
        X = self.X_(X, task)
        X = X[h]
        return (X, y)

    def fit(self, X, Y, task_features=None):
        X = np.asarray(X)
        Y = np.asarray(Y)
        self.ntasks = Y.shape[1]
        if task_features is None:
            task_features = [range(X.shape[1])] * self.ntasks
        self.task_features = task_features
        self.models = []
        for task in range(self.ntasks):
            Xt, yt = self.Xy_(X, Y, task)
            m = copy.deepcopy(self.model)
            # m = skb.clone(self.model)
            m.fit(Xt, yt)
            self.models.append(m)

    def predict(self, X):
        X = np.asarray(X)
        Y = []
        for task in range(self.ntasks):
            m = self.models[task]
            Xt = self.X_(X, task)
            Y.append(m.predict(Xt))
        Y = np.vstack(Y).T
        return Y

    def predict_proba(self, X):
        X = np.asarray(X)
        Y = []
        for task in range(self.ntasks):
            m = self.models[task]
            Xt = self.X_(X, task)
            Y.append(m.predict_proba(Xt)[:, 1])
        Y = np.vstack(Y).T
        return Y

    def get_params(self):
        return self.model.get_params()

    def set_params(self, **kwargs):
        return self.model.set_params(**kwargs)

    def coef_(self, order=False, abs_=True):
        fi = np.vstack([x.coef_ for x in self.models])
        if abs_:
            fi = np.abs(fi)
        if order:
            o = np.mean(fi, axis=0).argsort()
            return (fi, o)
        else:
            return fi

    def feature_importances_(self, order=False):
        fi = np.vstack([x.feature_importances_ for x in self.models])
        if order:
            o = np.mean(fi, axis=0).argsort()
            return (fi, o)
        else:
            return fi

    def feature_importances(self, *args, **kwargs):
        attr = None
        for a in ['feature_importances_', 'coef_']:
            if hasattr(self.models[0], a):
                attr = a
                break
        f = getattr(self, attr)
        return f(*args, **kwargs)


def sample_features(samples, features):
    """Return feature indices for different samples. Useful to create a
       sample specific classifier with MultitaskClassifier."""
    ffeatures = flatten_index(features)
    shared = []
    specific = []
    for i, f in enumerate(ffeatures):
        is_shared = True
        for s in samples:
            if f.find(s) >= 0:
                is_shared = False
                break
        if is_shared:
            shared.append(i)
        else:
            specific.append(i)
    rv = []
    for s in samples:
        sf = []
        for i in specific:
            if ffeatures[i].find(s) >= 0:
                sf.append(i)
        sf.extend(shared)
        rv.append(sf)
    return rv


def data_sample(X, Y, sample):
    y = Y.loc[:, sample]
    h = np.asarray(y.notnull())
    X = X.loc[h]
    y = y.loc[h]
    return (X, y)
