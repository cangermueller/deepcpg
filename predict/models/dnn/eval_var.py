#!/usr/bin/env python

import argparse
import sys
import logging
import os
import os.path as pt
import h5py as h5
import pandas as pd
import numpy as np

import predict.utils as ut
import predict.evaluation as ev


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
    t = ev.eval_to_str(p)
    with open(pt.join(out_dir, 'pred_%s.csv' % (name)), 'w') as f:
        f.write(t)


def evaluate(y, z):
    funs = [
        ('rmse', ev.rmse),
        ('mse', ev.mse),
        ('mad', ev.mad),
        ('cor', ev.cor)]
    e = ev.evaluate(y, z, funs=funs, mask=None)
    e['y'] = np.mean(y)
    e['z'] = np.mean(z)
    e['ymed'] = np.median(y)
    e['zmed'] = np.median(z)
    return e


def eval_global(y, z):
    return evaluate(y, z)


def eval_annos(y, z, chromos, cpos, annos_file, regexs=[r'loc_.+']):
    f = h5.File(annos_file)
    annos = list(f[chromos[0]].keys())
    f.close()
    annos = ut.filter_regex(annos, regexs)
    es = []
    for anno in annos:
        a = []
        for chromo, pos in zip(chromos, cpos):
            a.append(read_annos(annos_file, chromo, anno, pos)[1])
        a = np.hstack(a)
        if a.sum() > 10:
            e = evaluate(y[a], z[a])
            e['anno'] = anno
            es.append(e)
    es = pd.concat(es)
    es.sort_values('anno', inplace=True)
    return es


def eval_stats(y, z, chromos, cpos, stats_file, stats=None, nb_bin=5):
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

        while nb_bin > 0:
            try:
                bins = ut.qcut(ut.add_noise(s), nb_bin)
                break
            except ValueError:
                nb_bin -= 1
        if nb_bin == 0:
            raise ValueError('Insufficient observations for binning statistic!')

        for bin_ in bins.categories:
            t = bins == bin_
            e = evaluate(y[t], z[t])
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
            'pred_file',
            help='Prediction file')
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
            '--targets',
            help='Filter targets',
            nargs='+')
        p.add_argument(
            '--annos_file',
            help='HDF file with annotations')
        p.add_argument(
            '--annos',
            help='Regex of annotations to be considered',
            default=[r'^loc.+$', r'^licr_.+$'],
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

    def evaluate(self, target, sql_meta=dict()):
        log = self.log
        opts = self.opts

        sql_meta['target'] = target
        out_dir = opts.out_dir
        if out_dir is not None:
            out_dir = pt.join(out_dir, target)
            os.makedirs(out_dir, exist_ok=True)

        data = read_prediction(opts.pred_file, target=target,
                               chromos=opts.chromos,
                               what=['pos', 'y', 'z'])
        chromos, cpos = data['chromo'], data['pos']

        log.info('Evaluate global')
        e = evaluate(data['y'], data['z'])
        if opts.verbose:
            print('Global:')
            print(ev.eval_to_str(e))
        if out_dir is not None:
            write_output(e, 'global', out_dir)
        if opts.sql_file:
            ut.to_sql(opts.sql_file, e, 'global', sql_meta)

        if opts.annos_file is not None:
            log.info('Evaluate annos')
            ea = eval_annos(data['y'], data['z'], chromos, cpos,
                            opts.annos_file, regexs=opts.annos)
            if opts.verbose:
                print('Annos')
                print(ev.eval_to_str(ea))
            if out_dir is not None:
                write_output(ea, 'annos', out_dir)
            if opts.sql_file:
                ut.to_sql(opts.sql_file, ea, 'annos', sql_meta)

        if opts.stats_file is not None:
            log.info('Evaluate stats')
            es = eval_stats(data['y'], data['z'], chromos, cpos,
                            opts.stats_file, stats=opts.stats,
                            nb_bin=opts.stats_bins)
            if opts.verbose:
                print('Stats:')
                print(ev.eval_to_str(es))
            if out_dir is not None:
                write_output(es, 'stats', out_dir)
            if opts.sql_file:
                ut.to_sql(opts.sql_file, es, 'stats', sql_meta)

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
            sql_meta['path'] = pt.realpath(opts.pred_file)
            if opts.sql_meta is not None:
                for meta in opts.sql_meta:
                    k, v = meta.split('=')
                    sql_meta[k] = v

        log.info('Read')
        in_file = h5.File(opts.pred_file, 'r')
        targets = list(in_file.keys())
        if opts.targets:
            targets = sorted(ut.filter_regex(targets, opts.targets))
        for target in targets:
            log.info('Evaluate %s' % (target))
            self.evaluate(target, sql_meta)

        log.info('Done!')
        return 0

if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
