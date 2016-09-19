#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import h5py as h5
import pandas as pd
import numpy as np

import predict.utils as ut


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
    p = g['pos'].value
    d = [g[k].value for k in stats]
    f.close()
    d = np.vstack(d).T
    if pos is not None:
        t = np.in1d(p, pos)
        p = p[t]
        assert np.all(p == pos)
        d = d[t]
    return p, d


def evaluate(X, funs=[('mean', np.mean), ('median', np.median),
                      ('min', np.min), ('max', np.max)]):
    d = []
    for _, f in funs:
        d.append(f(X, axis=0))
    d = np.vstack(d).T
    d = pd.DataFrame(d, columns=[x for x, _ in funs], index=X.columns)
    d['n'] = X.shape[0]
    d.index.name = 'stat'
    d.reset_index(inplace=True)
    return d


def eval_global(X):
    return evaluate(X)


def eval_annos(X, chromos, pos, annos_file, regexs=[r'loc_.+'], nb_min=1):
    f = h5.File(annos_file)
    annos = list(f[chromos[0]].keys())
    f.close()
    annos = ut.filter_regex(annos, regexs)
    es = []
    for anno in annos:
        print(anno)
        a = []
        for c, chromo in enumerate(chromos):
            cpos = pos[c]
            o = cpos.argsort()
            ro = o.copy()
            ro[o] = np.arange(len(ro))
            ac = read_annos(annos_file, chromo, anno, cpos[o])
            assert np.all(ac[0][ro] == cpos)
            a.append(ac[1][ro])
        a = np.hstack(a)
        if a.sum() >= nb_min:
            e = evaluate(X.loc[a])
            e['anno'] = anno
            es.append(e)
    es = pd.concat(es)
    es.sort_values('anno', inplace=True)
    return es


def eval_to_str(x):
    return x.to_csv(None, sep='\t', index=False, float_format='%.4f')


def write_output(p, name, out_dir):
    t = eval_to_str(p)
    with open(pt.join(out_dir, '%s.csv' % (name)), 'w') as f:
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
            description='Evaluate statistics')
        p.add_argument(
            'in_file',
            help='Input file with statistics')
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
            '--sql_skip',
            help='Skip evaluation if entry exists in SQL file',
            action='store_true')
        p.add_argument(
            '--annos_file',
            help='HDF file with annotations')
        p.add_argument(
            '--annos',
            help='Regex of annotations to be considered',
            default=[r'^loc.+$'],
            nargs='+')
        p.add_argument(
            '--stats',
            help='Regex of statistics to be considered',
            default=[r'^.+$'],
            nargs='+')
        p.add_argument(
            '--chromos',
            help='Only apply to these chromosome',
            nargs='+')
        p.add_argument(
            '--nb_sample',
            help='Maximum # samples',
            type=int)
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

        self.log = log
        self.opts = opts

        sql_meta = dict()
        if opts.sql_file is not None:
            sql_meta['path'] = pt.realpath(opts.sql_file)
            if opts.sql_meta is not None:
                for meta in opts.sql_meta:
                    k, v = meta.split('=')
                    sql_meta[k] = v

        log.info('Read')
        in_file = h5.File(opts.in_file, 'r')
        chromos = opts.chromos
        if chromos is None:
            chromos = list(in_file.keys())
        stats = list(filter(lambda x: x != 'pos', in_file[chromos[0]].keys()))
        if opts.stats is not None:
            stats = ut.filter_regex(stats, opts.stats)
        nb_sample = in_file[chromos[0]]['pos'].shape[0]
        if opts.nb_sample is not None:
            nb_sample = min(nb_sample, opts.nb_sample)
        X = []
        pos = []
        for chromo in chromos:
            p, x = read_stats(opts.in_file, chromo, stats)
            pos.append(p)
            X.append(x)
        X = pd.DataFrame(np.vstack(X), columns=stats)
        in_file.close()

        log.info('Evaluate global')
        e = eval_global(X)
        if opts.verbose:
            print('Global statistics:')
            print(eval_to_str(e))
        if opts.out_dir is not None:
            write_output(e, 'global', opts.out_dir)
        if opts.sql_file is not None:
            ut.to_sql(opts.sql_file, e, 'global', sql_meta)

        if opts.annos_file is not None:
            log.info('Evaluate annos')
            e = eval_annos(X, chromos, pos, opts.annos_file, opts.annos)
            if opts.verbose:
                print('Annotation specific statistics:')
                print(eval_to_str(e))
            if opts.out_dir is not None:
                write_output(e, 'annos', opts.out_dir)
            if opts.sql_file is not None:
                ut.to_sql(opts.sql_file, e, 'annos', sql_meta)

        log.info('Done!')

        return 0

if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
