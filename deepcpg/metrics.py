from collections import OrderedDict

from keras import backend as K
from keras import metrics as kmet

from .utils import get_from_module


def contingency_table(y, z):
    z = K.round(z)

    def count_matches(a, b):
        tmp = K.concatenate([a, b])
        return K.sum(K.cast(K.all(tmp, -1), K.floatx()))

    ones = K.ones_like(y)
    zeros = K.zeros_like(y)
    y_ones = K.equal(y, ones)
    y_zeros = K.equal(y, zeros)
    z_ones = K.equal(z, ones)
    z_zeros = K.equal(z, zeros)

    tp = count_matches(y_ones, z_ones)
    tn = count_matches(y_zeros, z_zeros)
    fp = count_matches(y_zeros, z_ones)
    fn = count_matches(y_ones, z_zeros)

    return (tp, tn, fp, fn)


def acc(y, z):
    tp, tn, fp, fn = contingency_table(y, z)
    return (tp + tn) / (tp + tn + fn + fp)


def prec(y, z):
    tp, tn, fp, fn = contingency_table(y, z)
    return tp / (tp + fp)


def tpr(y, z):
    tp, tn, fp, fn = contingency_table(y, z)
    return tp / (tp + fn)


def tnr(y, z):
    tp, tn, fp, fn = contingency_table(y, z)
    return tn / (tn + fp)


def fpr(y, z):
    tp, tn, fp, fn = contingency_table(y, z)
    return fp / (fp + tn)


def fnr(y, z):
    tp, tn, fp, fn = contingency_table(y, z)
    return fn / (fn + tp)


def f1(y, z):
    _tpr = tpr(y, z)
    _prec = prec(y, z)
    return 2 * (_prec * _tpr) / (_prec + _tpr)


def mcc(y, z):
    tp, tn, fp, fn = contingency_table(y, z)
    return (tp * tn - fp * fn) /\
        K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))


def mse(y, z):
    return kmet.mean_squared_error(y, z)


def mae(y, z):
    return kmet.mean_absolute_error(y, z)


def get(name):
    return get_from_module(name, globals())
