import numpy as np
import sklearn.metrics as skm

def cor(y, z):
    return np.corrcoef(y, z)[0, 1]

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


eval_funs = [('auc', auc),
             ('acc', acc),
             ('tpr', tpr),
             ('tnr', tnr),
             ('mcc', mcc),
             ('rrmse', rrmse),
             ('cor', cor)]
