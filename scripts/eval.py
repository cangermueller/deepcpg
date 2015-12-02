#!/usr/bin/env python

import argparse
import sys
import logging
import os
import os.path as pt
import h5py as h5
import pandas as pd
import numpy as np
from predict.evaluation import evaluate, eval_to_str
import sqlite3 as sql
import hashlib


def to_sql(sql_path, data, table, meta):
    md5 = hashlib.md5()
    for v in sorted(meta.values()):
        md5.update(v.encode())
    id_ = md5.hexdigest()

    data = data.copy()
    for k, v in meta.items():
        data[k] = v
    data['id'] = id_
    con = sql.connect(sql_path)
    try:
        con.execute('DELETE FROM %s WHERE id = "%s"' % (table, id_))
    except sql.OperationalError:
        pass
    data.to_sql(table, con, if_exists='append', index=False)
    con.close()


def read_test(test_file, target=None, chromos=None):
    f = h5.File(test_file)
    if target is not None:
        g = f[target]
    else:
        g = f
    if chromos is None:
        chromos = list(g.keys())
    d = dict(pos=[], y=[], z=[])
    for chromo in chromos:
        for k in d.keys():
            d[k].append(g[chromo][k].value)
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
    annos_used = []
    for anno in annos:
        a = []
        for chromo, pos in zip(chromos, cpos):
            a.append(read_annos(annos_file, chromo, anno, pos)[1])
        a = np.hstack(a)
        if a.sum() > 10:
            ya = y[a]
            za = z[a]
            annos_used.append(anno)
            pa.append(evaluate(ya, za))
    pa = pd.concat(pa, axis=0)
    pa.index = pd.Index(annos_used)
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
            '--sql_meta',
            help='Meta columns in SQL table',
            nargs='+')
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

    def evaluate(self, sql_meta=dict(), target=None):
        log = self.log
        opts = self.opts

        if target is not None:
            sql_meta['target'] = target
        out_dir = opts.out_dir
        if out_dir is not None:
            out_dir = opts.out_dir
            if target is not None:
                out_dir = pt.join(out_dir, target)
                os.makedirs(out_dir, exist_ok=True)

        chromos, cpos, y, z = read_test(opts.test_file,
                                       target=target, chromos=opts.chromos)
        log.info('Evaluate global performance')
        p = evaluate(y, z)
        print('Global performance:')
        print(eval_to_str(p))
        if out_dir is not None:
            write_output(p, 'global', out_dir)
        if opts.sql_file:
            to_sql(opts.sql_file, p, 'global', sql_meta)

        if opts.annos_file is not None:
            log.info('Evaluate annotation-specific  performance')
            pa = eval_annos(y, z, chromos, cpos, opts.annos_file,
                            annos=opts.annos)
            print('Annotation-specific performance:')
            print(eval_to_str(pa))
            if out_dir is not None:
                write_output(pa, 'annos', out_dir)
            if opts.sql_file:
                to_sql(opts.sql_file, pa, 'annos', sql_meta)

        if opts.stats_file is not None:
            log.info('Evaluate statistics-based performance')
            ps = eval_stats(y, z, chromos, cpos, opts.stats_file,
                            stats=opts.stats,
                            nbins=opts.stats_bins)
            print('Statistics-based performance:')
            print(eval_to_str(ps))
            if out_dir is not None:
                write_output(ps, 'stats', out_dir)
            if opts.sql_file:
                to_sql(opts.sql_file, ps, 'stats', sql_meta)

    def main(self, name, opts):
        logging.basicConfig(filename=opts.log_file,
                            format='%(levelname)s (%(asctime)s): %(message)s')
        log = logging.getLogger(name)
        if opts.verbose:
            log.setLevel(logging.DEBUG)
        else:
            log.setLevel(logging.INFO)
        log.debug(opts)

        self.log = log
        self.opts = opts

        sql_meta = dict()
        if opts.sql_file is not None:
            sql_meta['path'] = pt.realpath(opts.test_file)
            if opts.sql_meta is not None:
                for meta in opts.sql_meta:
                    k, v = meta.split('=')
                    sql_meta[k] = v

        log.info('Read')
        f = h5.File(opts.test_file, 'r')
        if list(f.keys())[0].isdigit():
            self.evaluate(sql_meta)
        else:
            targets = list(f.keys())
            log.info('Found %d targets' % (len(targets)))
            for target in targets:
                log.info(target)
                self.evaluate(sql_meta, target)

        log.info('Done!')

        return 0

if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
