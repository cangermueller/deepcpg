import pandas as pd
import numpy as np
import sklearn.metrics as skm


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


def nll(y, z):
    eps = 1e-6
    y = y.ravel()
    z = z.ravel()
    t = y * np.log2(np.maximum(z, eps))
    t += (1 - y) * np.log2(np.maximum(1 - z, eps))
    t = t.sum() / len(t)
    return -t

eval_annos = ['misc_Active_enhancers', 'misc_CGI', 'misc_CGI_shelf',
              'misc_CGI_shore', 'misc_Exons', 'misc_H3K27ac', 'misc_H3K27me3',
              'misc_H3K4me1', 'misc_H3K4me1_Tet1', 'misc_IAP',
              'misc_Intergenic', 'misc_Introns', 'misc_LMRs', 'misc_Oct4_2i',
              'misc_TSSs', 'misc_gene_body', 'misc_mESC_enhancers', 'misc_p300',
              'misc_prom_2k05k', 'misc_prom_2k05k_cgi', 'misc_prom_2k05k_ncgi']

eval_funs = [('auc', auc),
             ('acc', acc),
             ('tpr', tpr),
             ('tnr', tnr),
             ('mcc', mcc),
             ('nll', nll),
             ('rrmse', rrmse),
             ('cor', cor)]

eval_funs_regress = [
    ('rmse', rmse),
    ('mse', mse),
    ('mad', mad),
    ('cor', cor)]


def evaluate(y, z, mask=-1, funs=eval_funs):
    y = y.ravel()
    z = z.ravel()
    if mask is not None:
        t = y != mask
        y = y[t]
        z = z[t]
    s = dict()
    for name, fun in funs:
        s[name] = fun(y, z)
    d = pd.DataFrame(s, columns=[x for x, _ in funs], index=[0])
    d['n'] = len(y)
    return d


def evaluate_all(y, z, *args, **kwargs):
    keys = sorted(z.keys())
    p = [evaluate(y[k][:], z[k][:], *args, **kwargs) for k in keys]
    p = pd.concat(p)
    p.index = keys
    return p


def eval_to_str(e, index=False, *args, **kwargs):
    s = e.to_csv(None, sep='\t', index=index, float_format='%.4f', *args,
                 **kwargs)
    return s


def eval_to_file(e, path, *args, **kwargs):
    with open(path, 'w') as f:
        f.write(eval_to_str(e, *args, **kwargs))
