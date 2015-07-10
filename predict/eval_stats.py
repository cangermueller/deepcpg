import argparse
import sys
import logging
import os.path as pt
import numpy as np
import pandas as pd
import warnings

from predict import utils as ut
from predict import hdf
from predict import data_select
from predict import data


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
    f = lambda x: __cpg_cov(x, mean=True)
    return ut.rolling_apply(x, delta, f).iloc[:, 0]


def __var(x, axis=0):
    """Return mean over axis and than variance."""
    return x.mean(axis=axis).var()


def __var_win(x, delta, axis=0):
    """Return variance in window."""
    f = lambda x: __var(x, axis=axis)
    return ut.rolling_apply(x, delta, f).iloc[:, 0]


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
    f = lambda x: __cpg_content(x, delta)
    return ut.rolling_apply(x, delta, f).iloc[:, 0]


def __cpg_density(x, delta):
    c = delta * x.shape[1]
    return x.notnull().sum().sum() / c


def cpg_density_win(x, delta):
    """Return density of CpG matrix in window."""
    f = lambda x: __cpg_density(x, delta)
    return ut.rolling_apply(x, delta, f).iloc[:, 0]


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


def __met_rate(x, delta):
    return x.mean(axis=0).mean()


def met_rate_win(x, delta):
    """Return mean methylation rate between samples in window."""
    f = lambda x: __met_rate(x, delta)
    return ut.rolling_apply(x, delta, f).iloc[:, 0]




class Processor(object):

    def __init__(self, in_path):
        self.in_path = in_path
        self.in_group = '/'
        self.out_path = '/es'
        self.out_group = '/'
        self.chromos = None
        self.logger = None

    def log(self, msg):
        if self.logger is not None:
            self.logger(msg)

    def process_chromo(self, stats, chromo):
        # Read combined train and val CpGs
        Y = data_select.select_cpg_matrix(self.in_path, self.in_group,
                                          chromos=[chromo],
                                          subsets=['train', 'val'])
        Y.index = Y.index.droplevel(0)
        for stat_name, stat_fun in stats.items():
            self.log(stat_name + ' ...')
            s = stat_fun(Y)
            assert type(s) is pd.Series
            assert np.all(s.index == Y.index)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                g = pt.join(self.out_group, stat_name, chromo)
                s.to_hdf(self.out_path, g)

    def process(self, stats):
        chromos = self.chromos
        if chromos is None:
            chromos = hdf.ls(self.in_path, pt.join(self.in_group, 'pos'))
        for chromo in chromos:
            self.log('Chromosome %s ...' % (str(chromo)))
            self.process_chromo(stats, str(chromo))


class Selector(object):

    def __init__(self, chromos=None, stats=None):
        self.chromos = chromos
        self.stats = stats

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
        if not isinstance(stats, list):
            stats = [stats]
        chromos = self.chromos
        if chromos is None:
            chromos = hdf.ls(path, pt.join(group, stats[0]))
        d = []
        for chromo in chromos:
            dc = self.select_chromo(path, group, chromo, stats)
            d.append(dc)
        d = pd.concat(d, keys=chromos, names=['chromo', 'pos'])
        return d
