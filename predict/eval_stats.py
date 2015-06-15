import argparse
import sys
import logging
import os.path as pt
import numpy as np
import pandas as pd
import warnings

from predict import utils as ut
from predict import hdf


def __cpg_cov(x, mean=False):
    """Return CpG coverage of matrix x."""
    h = np.mean(~np.isnan(x), axis=1)
    if mean:
        h = h.mean()
    return h


def cpg_cov(x):
    """Return CpG coverage of single CpGs"""
    return __cpg_cov(x)


def cpg_cov_win(x, delta):
    """Return mean CpG coverage in window."""
    return ut.rolling_apply(x, delta, __cpg_cov, mean=True).iloc[:, 0]


def __var(x, axis=0):
    """Return mean over axis and than variance."""
    x.mean(axis=axis).var()
    return x.mean(axis=axis).var()


def __var_win(x, delta, axis=0):
    """Return variance in window."""
    return ut.rolling_apply(x, delta, __var, axis=axis).iloc[:, 0]


def var_samples_win(x, delta):
    """Return variance between samples in window."""
    return __var_win(x, delta, axis=0)


def var_sites_win(x, delta):
    """Return variance between sites in window."""
    return __var_win(x, delta, axis=1)


def __cpg_content(x, delta):
    return x.shape[0] / delta


def cpg_content_win(x, delta):
    """Return CpG content (% CpG) in window."""
    return ut.rolling_apply(x, delta, __cpg_content, delta).iloc[:, 0]


def __cpg_density(x, delta):
    c = delta * x.shape[1]
    return x.notnull().sum().sum() / c


def cpg_density_win(x, delta):
    """Return density of CpG matrix in window."""
    return ut.rolling_apply(x, delta, __cpg_density, delta).iloc[:, 0]


def __min_dist_sample(x):
    """Returns minimal distance to nearest CpG for one sample.
    x is vector with location of covered CpGs.
    """
    rv = np.empty(len(x), dtype='int32')
    rv.fill(0)
    for i in range(len(x)):
        h = []
        if i > 0:
            h.append(abs(x[i] - x[i - 1]))
        if i < len(x) - 1:
            h.append(abs(x[i] - x[i + 1]))
        if len(h) > 0:
            rv[i] = min(h)
    return rv


def min_dist(x, fun=np.mean):
    """Return minimal distance of nearest covered CpG for all samples and than
    applies fun to these distances. Returns mean minimal distance to nearest
    neighbor by default."""
    h = []
    for i in range(x.shape[1]):
        xi = x.iloc[:, i].dropna()
        h.append(pd.DataFrame(__min_dist_sample(xi.index), index=xi.index))
    h = pd.concat(h, axis=1)
    h.columns = x.columns
    assert np.all(h.shape == x.shape)
    if fun is not None:
        h = fun(h, axis=1)
        assert h.notnull().all()
    return h


def __cor(x, axis=0, fun=np.mean):
    """Return mean correlation coefficient. Not used now."""
    if axis == 1:
        x = x.T
    xm = np.ma.masked_array(x, np.isnan(x))
    c = np.ma.corrcoef(xm).data
    c[c > 1.0] = 1.0
    c[c < -1.0] = -1.0
    if fun is not None:
        c = fun(c)
    return c


class Processor(object):

    def __init__(self, out_file):
        self.out_file = out_file
        self.chromos = None

    def process_chromo(self, in_file, stats, chromo):
        in_path, in_group = hdf.split_path(in_file)
        Y = pd.read_hdf(in_path, pt.join(in_group, 'Y', chromo))
        out_path, out_group = hdf.split_path(self.out_file)
        for stat_name, stat_fun in stats.items():
            print(stat_name)
            s = stat_fun(Y)
            assert type(s) is pd.Series
            assert np.all(s.index == Y.index)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                s.to_hdf(out_path, pt.join(out_group, stat_name, chromo))

    def process(self, in_file, stats):
        in_path, in_group = hdf.split_path(in_file)
        chromos = self.chromos
        if chromos is None:
            chromos = hdf.ls(in_path, pt.join(in_group, 'Y'))
        for chromo in chromos:
            self.process_chromo(in_file, stats, str(chromo))


class Selector(object):

    def __init__(self, chromos=None, stats=None):
        self.chromos = chromos
        self.stats = None

    def select_chromo(self, path, group, chromo, stats):
        d = []
        for stat in stats:
            g = pt.join(group, stat, chromo)
            ds = pd.read_hdf(path, g)
            d.append(ds)
        d = pd.concat(d, axis=1, keys=stats)
        return d

    def select(self, path, group):
        stats = self.stats
        if stats is None:
            stats = hdf.ls(path, group)
        chromos = self.chromos
        if chromos is None:
            chromos = hdf.ls(path, pt.join(group, stats[0]))
        d = []
        for chromo in chromos:
            dc = self.select_chromo(path, group, chromo, stats)
            d.append(dc)
        d = pd.concat(d, keys=chromos, names=['chromo', 'pos'])
        return d
