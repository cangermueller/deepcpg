#!/usr/bin/env python

import argparse
import sys
import logging
import os
import os.path as pt
import h5py as h5
import pandas as pd
import numpy as np
import sqlite3 as sql
import hashlib
import scipy.stats as sps

from predict.utils import filter_regex


def evaluate(x, funs=[('mean', np.mean), ('var', np.var)], axis=0):
    d = dict()
    for fn, f in funs:
        d[fn] = f(x, axis=axis)
    d = pd.DataFrame(d)
    d.index.name = 'effect'
    d = d.reset_index()
    d = pd.melt(d, id_vars='effect', var_name='fun', value_name='value')
    return d


def eval_to_str(x):
    h = list(x.columns[~x.columns.isin(['fun', 'value'])])
    x = pd.pivot_table(x, index=h, columns='fun', values='value')
    x = x.reset_index()
    x.columns.name = None
    return x.to_csv(None, sep='\t', index=False, float_format='%.4f')


def within_01(x, eps=1e-6):
    return np.maximum(eps, np.minimum(1 - eps, x))


def logodds(p, q):
    p = within_01(p)
    q = within_01(q)
    return np.log2(p) - np.log2(q)


def logodds_ratio(p, q):
    p = within_01(p)
    q = within_01(q)
    return np.log2(p / (1 - p)) - np.log2(q / (1 - q))


def seqmut_effect(z, zm, effect_size='abs'):
    if effect_size == 'del':
        return z - zm
    elif effect_size == 'abs':
        return np.abs(z - zm)
    elif effect_size == 'lor':
        return logodds_ratio(z, zm)
    elif effect_size == 'abs_lor':
        return np.abs(logodds_ratio(z, zm))
    else:
        raise ValueError('Effect type "%s" not supported!' % (effect_size))


def get_id(meta):
    md5 = hashlib.md5()
    for v in sorted(meta.values()):
        md5.update(v.encode())
    id_ = md5.hexdigest()
    return id_


def exits_meta(sql_path, table, meta):
    id_ = get_id(meta)
    con = sql.connect(sql_path)
    cmd = con.execute('SELECT id FROM %s WHERE id = "%s"' % (table, id_))
    count = len(cmd.fetchall())
    con.close()
    return count > 0


def to_sql(sql_path, data, table, meta):
    id_ = get_id(meta)

    data = data.copy()
    for k, v in meta.items():
        data[k] = v
    data['id'] = id_
    con = sql.connect(sql_path, timeout=999999)
    try:
        con.execute('DELETE FROM %s WHERE id = "%s"' % (table, id_))
        cols = pd.read_sql('SELECT * FROM %s LIMIT 1' % (table), con)
    except sql.OperationalError:
        cols = []
    if len(cols):
        t = sorted(set(data.columns) - set(cols.columns))
        if len(t):
            print('Ignoring columns %s' % (' '.join(t)))
            data = data.loc[:, cols.columns]
    data.to_sql(table, con, if_exists='append', index=False)
    con.close()


def read_prediction(path, target=None, chromos=None, what=['pos', 'z']):
    f = h5.File(path)
    if target is not None:
        g = f[target]
    else:
        g = f
    if chromos is None:
        chromos = list(g.keys())
    d = {x: [] for x in what}
    for chromo in chromos:
        for k in d.keys():
            d[k].append(g[chromo][k].value)
    f.close()
    for k in what:
        if k != 'pos':
            d[k] = np.hstack(d[k])
    d['chromo'] = chromos
    return d


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
    with open(pt.join(out_dir, 'seqmut_%s.csv' % (name)), 'w') as f:
        f.write(t)


def eval_annos(x, chromos, cpos, annos_file, regexs=[r'loc_.+']):
    f = h5.File(annos_file)
    annos = list(f[chromos[0]].keys())
    f.close()
    annos = filter_regex(annos, regexs)
    es = []
    for anno in annos:
        a = []
        for chromo, pos in zip(chromos, cpos):
            a.append(read_annos(annos_file, chromo, anno, pos)[1])
        a = np.hstack(a)
        if a.sum() > 10:
            e = []

            x1 = x.loc[a]
            x0 = x.loc[~a]
            e.append(evaluate(x1))
            t = evaluate(x0)
            t.fun += '0'
            e.append(t)

            tt = sps.ttest_ind(x0.values, x1.values)
            t = dict()
            t['statistic'] = tt.statistic
            t['pvalue'] = tt.pvalue
            t['effect'] = x.columns
            t = pd.DataFrame(t)
            t = pd.melt(t, id_vars='effect', var_name='fun', value_name='value')
            e.append(t)

            e = pd.concat(e)
            e['anno'] = anno
            es.append(e)
    es = pd.concat(es)
    es.sort_values('anno', inplace=True)
    return es


def add_noise(x, eps=1e-6):
    min_ = np.min(x)
    max_ = np.max(x)
    xeps = x + np.random.uniform(-eps, eps, len(x))
    xeps = np.maximum(min_, xeps)
    xeps = np.minimum(max_, xeps)
    return xeps


def qcut(x, nb_bins, *args, **kwargs):
    p = np.arange(0, 101, 100 / nb_bins)
    q = list(np.percentile(x, p))
    y = pd.cut(x, bins=q, include_lowest=True)
    assert len(y.categories) == nb_bins
    assert y.isnull().any() == False
    return y


def eval_stats(x, chromos, cpos, stats_file, stats=None, nbins=5):
    if stats is None:
        f = h5.File(stats_file)
        stats = f[chromos[0]].keys()
        f.close()
        stats = list(filter(lambda x: x != 'pos', stats))
    es = []
    for stat in stats:
        s = []
        for chromo, pos in zip(chromos, cpos):
            s.append(read_stats(stats_file, chromo, stat, pos)[1])
        s = np.hstack(s)

        while nbins > 0:
            try:
                bins = qcut(add_noise(s), nbins, precision=3)
                break
            except ValueError:
                nbins -= 1
        if nbins == 0:
            raise ValueError('Not enough observations for binning statistic!')

        for bin_ in bins.categories:
            t = bins == bin_
            e = evaluate(x[t])
            e['stat'] = stat
            e['bin'] = bin_
            es.append(e)
    es = pd.concat(es)
    es.sort_values(['stat', 'bin'], inplace=True)
    return es


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
            description='Evaluate effect of sequence mutations')
        p.add_argument(
            'seqmut_file',
            help='Sequence mutation file')
        p.add_argument(
            '-o', '--out_dir',
            help='Output directory')
        p.add_argument(
            '--seqmut_name',
            help='Name of sequence mutation',
            default='z_zero-1')
        p.add_argument(
            '-e', '--effect_sizes',
            help='Effect sizes',
            nargs='+',
            choices=['del', 'abs', 'lor', 'abs_lor'],
            default=['del', 'abs', 'lor', 'abs_lor']
        )
        p.add_argument(
            '--sql_file',
            help='Append to sqlite database')
        p.add_argument(
            '--sql_meta',
            help='Meta columns in SQL table',
            nargs='+')
        p.add_argument(
            '--sql_skip',
            help='Skip evaluation if entry exists in SQL file',
            action='store_true')
        p.add_argument(
            '--annos_file',
            help='HDF file with annotations')
        p.add_argument(
            '--annos',
            help='Regex of annotations to be considered',
            default=[r'^.+$'],
            nargs='+')
        p.add_argument(
            '--stats_file',
            help='HDF file with statistics')
        p.add_argument(
            '--stats',
            help='Statistics to be considered',
            default=['cov', 'var', 'entropy',
                     'win_cov', 'win_var', 'win_entropy', 'win_dist',
                     'gc_content', 'cg_obs_exp'],
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

        data = read_prediction(opts.seqmut_file, target=target,
                               chromos=opts.chromos,
                               what=['pos', 'z', opts.seqmut_name])
        chromos, cpos = data['chromo'], data['pos']
        z, zm = data['z'], data[opts.seqmut_name]
        effects = dict()
        for effect_size in opts.effect_sizes:
            effects[effect_size] = seqmut_effect(z, zm, effect_size)
        effects = pd.DataFrame(effects)

        skip = opts.sql_file is not None and opts.sql_skip

        if not skip or not exits_meta(opts.sql_file, 'global', sql_meta):
            log.info('Evaluate global effects')
            e = evaluate(effects)
            if opts.verbose:
                print('Global effects:')
                print(eval_to_str(e))
            if out_dir is not None:
                write_output(e, 'global', out_dir)
            if opts.sql_file:
                to_sql(opts.sql_file, e, 'global', sql_meta)

        if opts.annos_file is not None:
            if not skip or not exits_meta(opts.sql_file, 'annos', sql_meta):
                log.info('Evaluate annotation-specific effects')
                ea = eval_annos(effects, chromos, cpos, opts.annos_file,
                                regexs=opts.annos)
                if opts.verbose:
                    print('Annotation-specific effects:')
                    print(eval_to_str(ea))
                if out_dir is not None:
                    write_output(ea, 'annos', out_dir)
                if opts.sql_file:
                    to_sql(opts.sql_file, ea, 'annos', sql_meta)

        if opts.stats_file is not None:
            if not skip or not exits_meta(opts.sql_file, 'stats', sql_meta):
                log.info('Evaluate statistics-based effects')
                es = eval_stats(effects, chromos, cpos, opts.stats_file,
                                stats=opts.stats,
                                nbins=opts.stats_bins)
                if opts.verbose:
                    print('Statistics-based effects:')
                    print(eval_to_str(es))
                if out_dir is not None:
                    write_output(es, 'stats', out_dir)
                if opts.sql_file:
                    to_sql(opts.sql_file, es, 'stats', sql_meta)

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
            sql_meta['seqmut'] = opts.seqmut_name
            sql_meta['path'] = pt.realpath(opts.seqmut_file)
            if opts.sql_meta is not None:
                for meta in opts.sql_meta:
                    k, v = meta.split('=')
                    sql_meta[k] = v

        log.info('Read')
        f = h5.File(opts.seqmut_file, 'r')
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
