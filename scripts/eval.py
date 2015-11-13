#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import h5py as h5
import pandas as pd
import numpy as np
from predict.evaluation import evaluate, eval_to_str


def read_test(test_file, chromos=None):
    f = h5.File(test_file)
    if chromos is None:
        chromos = list(f.keys())
    d = dict(pos=[], y=[], z=[])
    for chromo in chromos:
        for k in d.keys():
            d[k].append(f[pt.join(chromo, k)].value)
    f.close()
    for k in ['y', 'z']:
        d[k] = np.hstack(d[k])
    return chromos, d['pos'], d['y'], d['z']

def read_annos(annos_file, chromo, name, pos=None):
    f = h5.File(annos_file)
    d = {k: f[pt.join(chromo, name, k)].value for k in ['pos', 'annos']}
    f.close()
    if pos is not None:
        t = np.in1d(d['pos'], pos)
        for k in d.keys():
            d[k] = d[k][t]
        assert np.all(d['pos'] == pos)
    d['annos'][d['annos'] >= 0] = 1
    d['annos'][d['annos'] < 0] = 0
    d['annos'] = d['annos'].astype('bool')
    return d['pos'], d['annos']

def read_stats(stats_file, chromo, stats, pos=None):
    f = h5.File(stats_file)
    g = f[chromo]
    d = {k: g[k].value for k in [stats, pos]}
    f.close()
    if pos is not None:
        t = np.in1d(d['pos'], pos)
        for k in d.keys():
            d[k] = d[k][t]
        assert np.all(d['pos'] == pos)
    return d['pos'], d[stats]

def write_output(p, name, out_dir):
    t = eval_to_str(p)
    with open(pt.join(out_dir, 'perf_%s.csv' % (name)), 'w') as f:
        f.write(t)

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
            description='Evaluates prediction')
        p.add_argument(
            'test_file',
            help='Test file')
        p.add_argument(
            '--annos_file',
            help='HDF file with annotations')
        p.add_argument(
            '--stats_file',
            help='HDF file with statistics')
        p.add_argument(
            '--chromos',
            help='Only consider these chromosomes',
            nargs='+')
        p.add_argument(
            '-o', '--out_dir',
            help='Output directory')
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

        out_dir = opts.out_dir
        if out_dir is None:
            out_dir = pt.dirname(opts.test_file)

        log.info('Read')
        chromos, cpos, y, z = read_test(opts.test_file, opts.chromos)

        log.info('Evaluate global performance')
        p = evaluate(y, z)
        print('Global performance:')
        print(eval_to_str(p))
        write_output(p, 'global', out_dir)

        if opts.annos_file is not None:
            log.info('Evaluate annotation-specific  performance')
            annos = ['Active_enhancers', 'CGI', 'H3K27ac', 'H3K4me1', 'LMRs']
            pa = []
            for anno in annos:
                a = []
                for chromo, pos in zip(chromos, cpos):
                    a.append(read_annos(opts.annos_file, chromo, anno, pos)[1])
                a = np.hstack(a)
                ya = y[a]
                za = z[a]
                pa.append(evaluate(ya, za))
            pa = pd.concat(pa, axis=0)
            pa.index = pd.Index(annos)
            pa.reset_index(inplace=True)
            pa.name = 'anno'
            print('Annotation-specific performance:')
            print(eval_to_str(pa))
            write_output(pa, 'annos', out_dir)

        if opts.stats_file is not None:
            log.info('Evaluate statistics-based performance')
            f = h5.File(opts.stats_file)
            stats = list(f[chromos[0]].keys())
            pa = []
            index = []
            for stat in stats:
                a = []
                for chromo, pos in zip(chromos, cpos):
                    a.append(read_stats(opts.stats_file, chromo, stat, pos)[1])
                a = np.hstack(a)
                bins = pd.qcut(pa, bins=5)
                for b in bins.categories:
                    t = bins == b
                    ya = y[t]
                    za = y[t]
                    pa.append(evaluate(ya, za))
                    index.append((stat, b))
            pa = pd.concat(pa, axis=0)
            pa.index = pd.MultiIndex.from_tuple(index)
            pa.index.names = ['anno', 'name']
            pa.reset_index(inplace=True)
            print('Statistics-based performance:')
            print(eval_to_str(pa))
            write_output(pa, 'stats', out_dir)

        return 0

if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
