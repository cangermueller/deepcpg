#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import h5py as h5
import numpy as np
import warnings
from predict.evaluation import evaluate, eval_to_str

def read_cpg(data_file, chromo, max_samples):
    f = h5.File(data_file)
    d = dict()
    for k in ['cpg', 'pos']:
        ds = f[pt.join('cpg', chromo, k)]
        if max_samples is None:
            d[k] = ds.value
        else:
            d[k] = ds[:max_samples]
    return d['pos'], d['cpg']

def read_knn(data_file, chromo, knn_group, pos=None):
    f = h5.File(data_file)
    d = dict()
    for k in ['knn', 'pos', 'dist']:
        ds = f[pt.join(knn_group, chromo, k)]
        d[k] = ds.value
    if pos is not None:
        t = np.in1d(d['pos'], pos)
        d['pos'] = d['pos'][t]
        assert np.all(d['pos'] == pos)
        for k in ['dist', 'knn']:
            d[k] = d[k][t]
    return d['pos'], d['knn'], d['dist']

def read_data(data_file, chromo, knn_group, max_samples=None):
    pos, cpg = read_cpg(data_file, chromo, max_samples)
    *_, knn, dist = read_knn(data_file, chromo, knn_group, pos)
    return dict(pos=pos, y=cpg, knn=knn, dist=dist)

def write_data(data_file, chromo, d):
    f = h5.File(data_file, 'a')
    if chromo in f:
        del f[chromo]
    for k in ['pos', 'y', 'z']:
        f[pt.join(chromo, k)] = d[k]
    f.close()


def win_avg(knn, dist, delta=1500):
    x = knn.astype('float32')
    x[dist > delta] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        z = np.nanmean(x, axis=1)
    z[np.isnan(z)] = knn.mean()
    assert np.isnan(z).sum() == 0
    return z

class App(object):

    def run(self, args):
        name = pt.basename(args[0])
        parser = self.create_parser(name)
        opts = parser.parse_args(args[1:])
        return self.main(name, opts)

    def create_parser(self, name):
        p = argparse.ArgumentParser(
            prog=name,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Imputes by window averaging')
        p.add_argument(
            'data_file',
            help='HDF path to data')
        p.add_argument(
            '-o', '--out_file',
            help='Output file')
        p.add_argument(
            '--chromos',
            help='Chromosomes',
            nargs='+')
        p.add_argument(
            '-d', '--delta',
            help='Maximum distance',
            type=int,
            default=1500)
        p.add_argument(
            '--knn_group',
            help='Name of KNN group',
            default='knn_shared')
        p.add_argument(
            '--max_samples',
            help='Maximum number of samples',
            type=int)
        p.add_argument(
            '--seed',
            help='Seed of rng',
            type=int,
            default=0)
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

        if opts.seed is not None:
            np.random.seed(opts.seed)

        chromos = opts.chromos
        if chromos is None:
            f = h5.File(opts.data_file)
            chromos = list(f['cpg'].keys())
            f.close()
        for chromo in chromos:
            log.info('Chromosome %s' % (chromo))
            log.info('Read data')
            d = read_data(opts.data_file, chromo, opts.knn_group, opts.max_samples)
            log.info('Window average')
            d['z'] = win_avg(d['knn'], d['dist'], opts.delta)
            p = eval_to_str(evaluate(d['y'], d['z']))
            print('Performance:')
            print(p)

            log.info('Write data')
            write_data(opts.out_file, chromo, d)
        log.info('Done!')

        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
