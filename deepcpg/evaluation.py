from collections import OrderedDict

import numpy as np
import sklearn.metrics as skm

from .data import CPG_NAN


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


def auc(y, z, round=True):
    if round:
        y = y.round()
    if len(y) == 0 or len(np.unique(y)) < 2:
        return np.nan
    return skm.roc_auc_score(y, z)


def acc(y, z, round=True):
    if round:
        y = np.round(y)
        z = np.round(z)
    return skm.accuracy_score(y, z)


def tpr(y, z, round=True):
    if round:
        y = np.round(y)
        z = np.round(z)
    return skm.recall_score(y, z)


def tnr(y, z, round=True):
    if round:
        y = np.round(y)
        z = np.round(z)
    c = skm.confusion_matrix(y, z)
    return c[0, 0] / c[0].sum()


def mcc(y, z, round=True):
    if round:
        y = np.round(y)
        z = np.round(z)
    return skm.matthews_corrcoef(y, z)


def f1(y, z, round=True):
    if round:
        y = np.round(y)
        z = np.round(z)
    return skm.f1_score(y, z)


CLA_METRICS = [auc, acc, tpr, tnr, f1, mcc]

REG_METRICS = [mse, mad, cor]


def evaluate(y, z, mask=CPG_NAN, metrics=CLA_METRICS):
    y = y.ravel()
    z = z.ravel()
    if mask is not None:
        t = y != mask
        y = y[t]
        z = z[t]
    p = OrderedDict()
    for metric in metrics:
        p[metric.__name__] = metric(y, z)
    p['n'] = len(y)
    return p
