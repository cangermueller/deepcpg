from collections import OrderedDict

import numpy as np
import sklearn.metrics as skm

from .data.preprocess import CPG_NAN



def cor(y, z):
    return np.corrcoef(y, z)[0, 1]


def mad(y, z):
    return np.mean(np.abs(y - z))


def mse(y, z):
    return np.mean((y - z)**2)


def rmse(y, z):
    return np.sqrt(mse(y, z))


def rrmse(y, z):
    return 1 - rmse(y, z)


def auc(y, z):
    if len(y) == 0 or len(np.unique(y)) < 2:
        return np.nan
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


def f1(y, z, r=True):
    if r:
        z = np.round(z)
    return skm.f1_score(y, z)


def nll(y, z):
    eps = 1e-6
    y = y.ravel()
    z = z.ravel()
    t = y * np.log2(np.maximum(z, eps))
    t += (1 - y) * np.log2(np.maximum(1 - z, eps))
    t = t.sum() / len(t)
    return -t

CLASS_METRICS = OrderedDict([('auc', auc),
                             ('acc', acc),
                             ('tpr', tpr),
                             ('tnr', tnr),
                             ('f1', f1),
                             ('mcc', mcc),
                             ('cor', cor)])

REGR_METRICS = OrderedDict([('rmse', rmse),
                            ('mse', mse),
                            ('mad', mad),
                            ('cor', cor)])


def evaluate(y, z, mask=CPG_NAN, metrics=CLASS_METRICS):
    y = y.ravel()
    z = z.ravel()
    if mask is not None:
        t = y != mask
        y = y[t]
        z = z[t]
    p = OrderedDict()
    for metric, fun in metrics.items():
        p[metric] = fun(y, z)
    p['n'] = len(y)
    return p
