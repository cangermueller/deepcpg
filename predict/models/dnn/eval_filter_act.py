#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import h5py as h5
import pandas as pd
import numpy as np
import scipy.stats as sps

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
    d = {k: g[k].value for k in [stats, 'pos']}
    f.close()
    if pos is not None:
        t = np.in1d(d['pos'], pos)
        for k in d.keys():
            d[k] = d[k][t]
        assert np.all(d['pos'] == pos)
    return d['pos'], d[stats]


def evaluate(act, z, filts, targets):
    stats = []
    for f, filt in enumerate(filts):
        act_filt = act[:, f]
        act_mean = act_filt.mean()
        act_std = act_filt.std()
        for t, target in enumerate(targets):
            s = []
            s.append(filt)
            s.append(target)
            s.append(act_filt.shape[0])
            r = sps.spearmanr(act_filt, z[:, t])
            s.append(r[0])
            s.append(r[1])
            s.append(act_mean)
            s.append(act_std)
            stats.append(s)
    stats = pd.DataFrame(stats, columns=['filt', 'target', 'n', 'r', 'r_pvalue',
                                         'act_mean', 'act_std'])
    return stats


def eval_global(act, z, filts, targets):
    return evaluate(act, z, filts, targets)


def eval_annos(act, z, filts, targets, chromos, pos, annos_file,
               regexs=[r'loc_.+'], n_min=10):
    f = h5.File(annos_file)
    annos = list(f[chromos[0]].keys())
    f.close()
    annos = ut.filter_regex(annos, regexs)
    es = []
    uchromos = np.unique(chromos)
    for anno in annos:
        a = []
        for chromo in uchromos:
            t = chromos == chromo
            cpos = pos[t]
            o = cpos.argsort()
            ro = o.copy()
            ro[o] = np.arange(len(ro))
            ac = read_annos(annos_file, chromo, anno, cpos[o])
            assert np.all(ac[0][ro] == cpos)
            a.append(ac[1][ro])
        a = np.hstack(a)
        if a.sum() >= n_min:
            e = evaluate(act[a], z[a], filts, targets)
            e['anno'] = anno
            es.append(e)
    es = pd.concat(es)
    es.sort_values('anno', inplace=True)
    return es


def eval_stats(act, z, filts, targets, chromos, pos, stats_file, stats=None,
               nb_bin=5):
    if stats is None:
        f = h5.File(stats_file)
        stats = f[chromos[0]].keys()
        f.close()
        stats = list(filter(lambda x: x != 'pos', stats))
    es = []
    uchromos = np.unique(chromos)
    for stat in stats:
        s = []
        for chromo in uchromos:
            cpos = pos[chromos == chromo]
            o = cpos.argsort()
            ro = o.copy()
            ro[o] = np.arange(len(ro))
            cs = read_stats(stats_file, chromo, stat, cpos[o])
            assert np.all(cs[0][ro] == cpos)
            s.append(cs[1][ro])
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
            e = evaluate(act[t], z[t], filts, targets)
            e['stat'] = stat
            e['bin'] = bin_
            es.append(e)
    es = pd.concat(es)
    es.sort_values(['stat', 'bin'], inplace=True)
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
            description='Evaluate effect of sequence mutations')
        p.add_argument(
            'filter_act_file',
            help='Filter activation file')
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
            '--act_op',
            help='Operation applied to filter activations',
            choices=['mean', 'max'],
            default='mean')
        p.add_argument(
            '--nb_sample',
            help='Maximum # samples',
            type=int)
        p.add_argument(
            '--filters',
            help='Filters to be tested',
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
        in_file = h5.File(opts.filter_act_file, 'r')
        nb_sample = in_file['pos'].shape[0]
        if opts.nb_sample is not None:
            nb_sample = min(nb_sample, opts.nb_sample)
        data = dict()
        for x in ['chromo', 'pos', 'targets', 'z']:
            data[x] = in_file[x][:nb_sample]
        for x in ['chromo', 'targets']:
            data[x] = np.array([y.decode() for y in data[x]])

        chunk_size = int(1e6)
        nb_filt = in_file['act'].shape[2]
        filts = opts.filters
        if filts is None:
            filts = range(nb_filt)
        else:
            filts = ut.ranges_to_list(filts, 0, nb_filt - 1)
        nb_filt = len(filts)
        data['act'] = np.empty((nb_sample, nb_filt), dtype=in_file['act'].dtype)
        for i in range(0, nb_sample, chunk_size):
            s = slice(i, min(nb_sample, i + chunk_size))
            t = in_file['act'][s, :, filts]
            if opts.act_op == 'mean':
                t = t.mean(axis=1)
            else:
                t = t.max(axis=1)
            data['act'][s] = t

        print(data['act'].shape)
        print(nb_sample)
        log.info('Evaluate global')
        e = eval_global(data['act'], data['z'], filts, data['targets'])
        if opts.verbose:
            print('Global statistics:')
            print(eval_to_str(e))
        if opts.out_dir is not None:
            write_output(e, 'global', opts.out_dir)
        if opts.sql_file is not None:
            ut.to_sql(opts.sql_file, e, 'global', sql_meta)

        if opts.annos_file is not None:
            log.info('Evaluate annos')
            e = eval_annos(data['act'], data['z'], filts, data['targets'],
                           data['chromo'], data['pos'],
                           opts.annos_file, opts.annos)
            if opts.verbose:
                print('Annotation specific statistics:')
                print(eval_to_str(e))
            if opts.out_dir is not None:
                write_output(e, 'annos', opts.out_dir)
            if opts.sql_file is not None:
                ut.to_sql(opts.sql_file, e, 'annos', sql_meta)

        if opts.stats_file is not None:
            log.info('Evaluate stats')
            e = eval_stats(data['act'], data['z'], filts, data['targets'],
                           data['chromo'], data['pos'],
                           opts.stats_file, opts.stats, opts.stats_bins)
            if opts.verbose:
                print('Statistics:')
                print(eval_to_str(e))
            if opts.out_dir is not None:
                write_output(e, 'stats', opts.out_dir)
            if opts.sql_file is not None:
                ut.to_sql(opts.sql_file, e, 'stats', sql_meta)

        in_file.close()
        log.info('Done!')

        return 0

if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
