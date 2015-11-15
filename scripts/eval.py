#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import h5py as h5
import pandas as pd
import numpy as np
from predict.evaluation import evaluate, eval_to_str
import sqlite3 as sql


def to_sql(sql_path, data, table, meta):
    data = data.copy()
    for k, v in meta.items():
        data[k] = v
    con =  sql.connect(sql_path)
    data.to_sql(table, con, if_exists='append', index=False)
    con.close()

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
    d = {k: g[k].value for k in [stats, 'pos']}
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

def eval_annos(y, z, chromos, cpos, annos_file, annos=None):
    if annos is None:
        f = h5.File(annos_file)
        annos = list(f[chromos[0]].keys())
        f.close()
        annos = list(filter(lambda x: x.startswith('loc_'), annos))
    pa = []
    for anno in annos:
        a = []
        for chromo, pos in zip(chromos, cpos):
            a.append(read_annos(annos_file, chromo, anno, pos)[1])
        a = np.hstack(a)
        ya = y[a]
        za = z[a]
        pa.append(evaluate(ya, za))
    pa = pd.concat(pa, axis=0)
    pa.index = pd.Index(annos)
    pa.index.name = 'anno'
    pa.reset_index(inplace=True)
    pa.sort_values('anno', inplace=True)
    return pa

def eval_stats(y, z, chromos, cpos, stats_file, stats=None, nbins=5):
    if stats is None:
        f = h5.File(stats_file)
        stats = f[chromos[0]].keys()
        f.close()
        stats = list(filter(lambda x: x != 'pos', stats))
    ps = []
    index = []
    for stat in stats:
        s = []
        for chromo, pos in zip(chromos, cpos):
            s.append(read_stats(stats_file, chromo, stat, pos)[1])
        s = np.hstack(s)
        while nbins > 0:
            try:
                bins = pd.qcut(s, q=nbins, precision=3)
                break
            except ValueError:
                nbins -= 1
        if nbins == 0:
            raise ValueError('Not enough observations for binning statistic!')

        for bin_ in bins.categories:
            t = bins == bin_
            ys = y[t]
            zs = z[t]
            ps.append(evaluate(ys, zs))
            index.append((stat, bin_))
    ps = pd.concat(ps, axis=0)
    ps.index = pd.MultiIndex.from_tuples(index)
    ps.index.names = ['stat', 'bin']
    ps.reset_index(inplace=True)
    ps.sort_values(['stat', 'bin'], inplace=True)
    return ps

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
            '-o', '--out_dir',
            help='Output directory')
        p.add_argument(
            '--sql_file',
            help='Append to sqlite database')
        p.add_argument(
            '--sql_label',
            help='Label for sqlite database')
        p.add_argument(
            '--annos_file',
            help='HDF file with annotations')
        p.add_argument(
            '--annos',
            help='Annotations to be considered',
            nargs='+')
        p.add_argument(
            '--stats_file',
            help='HDF file with statistics')
        p.add_argument(
            '--stats',
            help='Statistics to be considered',
            default=['cov', 'var', 'entropy',
                     'win_cov', 'win_var', 'win_entropy', 'win_dist'],
            nargs='+')
        p.add_argument(
            '--stats_bins',
            help='Number of bins of quantization',
            type=int,
            default=5)
        p.add_argument(
            '--chromos',
            help='Only consider these chromosomes',
            nargs='+')
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

        if opts.sql_file is not None:
            assert opts.sql_label is not None
            sql = dict()
            sql['label'] = opts.sql_label
            sql['path'] = pt.realpath(opts.test_file)
            sql['sample'] = pt.basename(pt.dirname(pt.dirname(opts.test_file)))

        log.info('Read')
        chromos, cpos, y, z = read_test(opts.test_file, opts.chromos)

        log.info('Evaluate global performance')
        p = evaluate(y, z)
        print('Global performance:')
        print(eval_to_str(p))
        write_output(p, 'global', out_dir)
        if opts.sql_file:
            to_sql(opts.sql_file, p, 'global', sql)

        if opts.annos_file is not None:
            log.info('Evaluate annotation-specific  performance')
            pa = eval_annos(y, z, chromos, cpos, opts.annos_file,
                            annos=opts.annos)
            print('Annotation-specific performance:')
            print(eval_to_str(pa))
            write_output(pa, 'annos', out_dir)
            if opts.sql_file:
                to_sql(opts.sql_file, pa, 'annos', sql)

        if opts.stats_file is not None:
            log.info('Evaluate statistics-based performance')
            ps = eval_stats(y, z, chromos, cpos, opts.stats_file,
                            stats=opts.stats,
                            nbins=opts.stats_bins)
            print('Statistics-based performance:')
            print(eval_to_str(ps))
            write_output(ps, 'stats', out_dir)
            if opts.sql_file:
                to_sql(opts.sql_file, ps, 'stats', sql)

        log.info('Done!')

        return 0

if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
