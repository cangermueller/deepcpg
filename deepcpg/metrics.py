"""Functions for computing performance metrics during training using Keras.

Similar to :module:`evaluation`, but uses Keras tensors instead of numpy arrays
as input.
"""
from __future__ import division
from __future__ import print_function

from keras import backend as K

from .utils import get_from_module
from .data import CPG_NAN


def contingency_table(y, z):
    """Compute contingency table."""
    y = K.round(y)
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


def prec(y, z):
    """Compute precision."""
    tp, tn, fp, fn = contingency_table(y, z)
    return tp / (tp + fp)


def tpr(y, z):
    """Compute true positive rate."""
    tp, tn, fp, fn = contingency_table(y, z)
    return tp / (tp + fn)


def tnr(y, z):
    """Compute true negative rate."""
    tp, tn, fp, fn = contingency_table(y, z)
    return tn / (tn + fp)


def fpr(y, z):
    """Compute false positive rate."""
    tp, tn, fp, fn = contingency_table(y, z)
    return fp / (fp + tn)


def fnr(y, z):
    """Compute false negative rate."""
    tp, tn, fp, fn = contingency_table(y, z)
    return fn / (fn + tp)


def f1(y, z):
    """Compute F1 score."""
    _tpr = tpr(y, z)
    _prec = prec(y, z)
    return 2 * (_prec * _tpr) / (_prec + _tpr)


def mcc(y, z):
    """Compute Matthew's correlation coefficient."""
    tp, tn, fp, fn = contingency_table(y, z)
    return (tp * tn - fp * fn) /\
        K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))


def acc(y, z):
    """Compute accuracy."""
    tp, tn, fp, fn = contingency_table(y, z)
    return (tp + tn) / (tp + tn + fp + fn)


def _sample_weights(y, mask=None):
    """Compute sample weights."""
    if mask is None:
        weights = K.ones_like(y)
    else:
        weights = 1 - K.cast(K.equal(y, mask), K.floatx())
    return weights


def _cat_sample_weights(y, mask=None):
    return 1 - K.cast(K.equal(K.sum(y, axis=-1), 0), K.floatx())


def cat_acc(y, z):
    """Compute categorical accuracy given one-hot matrices."""
    weights = _cat_sample_weights(y)
    _acc = K.cast(K.equal(K.argmax(y, axis=-1),
                          K.argmax(z, axis=-1)),
                  K.floatx())
    _acc = K.sum(_acc * weights) / K.sum(weights)
    return _acc


def mse(y, z, mask=CPG_NAN):
    """Compute mean squared error."""
    weights = _sample_weights(y, mask)
    _mse = K.sum(K.square(y - z) * weights) / K.sum(weights)
    return _mse


def mae(y, z, mask=CPG_NAN):
    """Compute mean absolute deviation."""
    weights = _sample_weights(y, mask)
    _mae = K.sum(K.abs(y - z) * weights) / K.sum(weights)
    return _mae


def get(name):
    """Return object from module by its name."""
    return get_from_module(name, globals())
