"""Functions for computing statistic about binary CpG matrix.

CpG matrix x assumed to have shape
    * [sites, cells] for per CpG statistics
    * [sites, cells, context] for window-based statistics
"""

from __future__ import division
from __future__ import print_function

import numpy as np

from ..utils import EPS, get_from_module


def mean(x):
    """Mean methylation rate."""
    if x.ndim > 2:
        x = x.mean(axis=2)
    return np.mean(x, 1)


def mode(x):
    """Mode of methylation rate."""
    if x.ndim > 2:
        x = x.mean(axis=2)
    return x.mean(axis=1).round().astype(np.int8)


def var(x, *args, **kwargs):
    """Variance between cells."""
    if x.ndim > 2:
        x = x.mean(axis=2)
    return x.var(axis=1)


def cat_var(x, nb_bin=3, *args, **kwargs):
    """Categorical variance between cells.

    Discretizes variance from :func:`var` into `nb_bin` equally-spaced bins.
    """
    v = var(x, *args, **kwargs)
    bins = np.linspace(-EPS, 0.25, nb_bin + 1)
    cv = np.digitize(v, bins, right=True) - 1
    return np.ma.masked_array(cv, v.mask)


def cat2_var(*args, **kwargs):
    """Binary variance between cells."""
    cv = cat_var(*args, **kwargs)
    cv[cv > 0] = 1
    return cv


def entropy(x):
    """Entropy of single CpG sites between cells."""
    if x.ndim > 2:
        x = x.mean(axis=2)
    p1 = x.mean(axis=1)
    p1 = np.minimum(1 - EPS, np.maximum(EPS, p1))
    p0 = 1 - p1
    return -(p1 * np.log(p1) + p0 * np.log(p0))


def diff(x):
    """Test if CpG site is differentially methylated."""
    if x.ndim > 2:
        x = x.mean(axis=2)
    return x.min(axis=1) != x.max(axis=1).astype(np.int8)


def get(name):
    """Return object from module by its name."""
    return get_from_module(name, globals())
