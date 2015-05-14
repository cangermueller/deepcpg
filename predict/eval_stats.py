#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import numpy as np
import pandas as pd


__dir = pt.dirname(pt.realpath(__file__))
# sys.path.insert(0, pt.join(__dir, '../module'))

import data as dat
import utils as ut
import hdf


def cpg_cov(x, mean=False):
    """Returns CpG coverage at position f(x) = # covererd / # samples"""

    h = np.mean(~np.isnan(x), axis=1)
    if mean:
        h = h.mean()
    return h


def cpg_density(x, c=1):
    """Returns CpG densitity f(x) = # CpGs in in window of width c / c"""

    return x.shape[0] / c


def var(x):
    """Returns variance \
        f(x) = variance(mean methylation in window of each sample)"""

    x = np.ma.masked_array(x, np.isnan(x))
    return x.mean(axis=0).var()


def min_dist_sample(x):
    """Returns minimal distance to nearest CpG for one sample.
    x is vector with location of covered CpGs.
    """

    rv = np.empty(len(x))
    rv.fill(np.nan)
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
    """Computes minimal distance of nearest covered CpG for all samples and than
    applies fun to these distances. Returns mean minimal distance to nearest
    neighbor by default."""

    h = []
    for i in range(x.shape[1]):
        xi = x.iloc[:, i].dropna()
        h.append(pd.DataFrame(min_dist_sample(xi.index), index=xi.index))
    h = pd.concat(h, axis=1)
    h.columns = x.columns
    assert np.all(h.shape == x.shape)
    if fun is not None:
        h = np.ma.masked_array(h, np.isnan(h))
        h = fun(h, axis=1).data
        h[h == 0] = np.nan
    return h


def stats(x, delta=1500):
    """Returns statistic of window of size 2*delta+1 centered on each CpG."""

    s = []
    s.append(cpg_cov(x))
    s.append(min_dist(x))
    s.append(ut.rolling_apply(x, delta, cpg_cov, mean=True))
    s.append(ut.rolling_apply(x, delta, cpg_density, c=(2 * delta + 1)))
    s.append(ut.rolling_apply(x, delta, var))
    s = [pd.DataFrame(s_, index=x.index) for s_ in s]
    s = pd.concat(s, axis=1)
    s.columns = ['cpg_cov', 'min_dist', 'win_cpg_cov',
                 'win_cpg_density', 'win_var']
    assert s.shape[0] == x.shape[0]
    return s


def stats_all(d, *args, **kwargs):
    def set_index(d):
        d.index = d.index.get_level_values(1)
        return d
    return ut.group_apply(d, 'chromo', lambda x: stats(set_index(x), *args, **kwargs),
                          level=True, set_index=True)


class EvalStats(object):

    def run(self, args):
        name = pt.basename(args[0])
        parser = self.create_parser(name)
        opts = parser.parse_args(args[1:])
        return self.main(name, opts)

    def create_parser(self, name):
        p = argparse.ArgumentParser(
            prog=name,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Computes evaluation statistics')
        p.add_argument(
            'in_file',
            help='HDF path of data set')
        p.add_argument('-o', '--out_file',
                       help='HDF path of output statistics',
                       default='eval_stats.h5')
        p.add_argument('--out_csv',
                       help='Write output statistics to CSV file')
        p.add_argument(
            '--verbose',
            help='More detailed log messages',
            action='store_true')
        p.add_argument(
            '--log_file',
            help='Write log messages to file')
        return p

    def main(self, name, opts):
        logging.basicConfig(filename=opts.log_file,
                            format='%(levelname)s (%(asctime)s): %(message)s')
        log = logging.getLogger(name)
        if opts.verbose:
            log.setLevel(logging.DEBUG)
        else:

            log.setLevel(logging.INFO)
        log.debug(opts)

        log.info('Read input ...')
        hdf_file, hdf_path = hdf.split_path(opts.in_file)
        d = pd.read_hdf(hdf_file, hdf_path)
        d = d.query('feature == "cpg" | sample == "global"')

        log.info('Compute statistics ...')
        Y = d.loc[d.feature == 'cpg']
        Y = dat.feature_matrix(d)
        Y.columns = Y.columns.get_level_values(0)
        S = stats_all(Y).reset_index()
        S = pd.melt(S, id_vars=['chromo', 'pos'],
                     var_name='feature', value_name='value')
        S['sample'] = 'stats'

        log.info('Write output ...')
        d = pd.concat([d, S])
        hdf_file, hdf_path = hdf.split_path(opts.out_file)
        hdf_path = pt.join(hdf_path, 'stats')
        log.info('\t%s' % (hdf_file))
        d.to_hdf(hdf_file, hdf_path)
        if opts.out_csv:
            log.info('\t%s' % (opts.out_csv))
            d.to_csv(opts.out_csv, index=False)

        log.info('Done!')
        return 0


if __name__ == '__main__':
    app = EvalStats()
    app.run(sys.argv)
