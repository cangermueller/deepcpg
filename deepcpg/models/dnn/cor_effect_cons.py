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
import predict.io as io


def evaluate(x, y, x_names, y_names):
    # x, y should be float64!
    stats = []
    for i in range(x.shape[1]):
        x_mean = x[:, i].mean()
        x_std = x[:, i].std()
        for j in range(y.shape[1]):
            s = pd.Series()
            s['x_name'] = x_names[i]
            s['y_name'] = y_names[j]
            xx = x[:, i]
            yy = y[:, j]
            _ = ~(np.isnan(xx) | np.isnan(yy))
            xx = xx[_]
            yy = yy[_]
            s['n'] = len(xx)
            r = sps.pearsonr(xx, yy)
            s['rp'] = r[0]
            s['rp_pvalue'] = r[1]
            r = sps.spearmanr(xx, yy)
            s['rs'] = r[0]
            s['rs_pvalue'] = r[1]
            s['x_mean'] = x_mean
            s['x_std'] = x_std
            s['y_mean'] = yy.mean()
            s['y_std'] = yy.std()
            stats.append(s)
    stats = pd.DataFrame(stats)
    return stats


def eval_global(x, y, x_names, y_names):
    return evaluate(x, y, x_names, y_names)


def eval_annos(x, y, x_names, y_names, chromos, cpos, annos_file,
               regexs=[r'loc_.+'], n_min=10):
    f = h5.File(annos_file)
    annos = list(f[chromos[0]].keys())
    f.close()
    annos = ut.filter_regex(annos, regexs)
    es = []
    for anno in annos:
        _ = io.read_annos(annos_file, chromos=chromos, pos=cpos,
                          regex='^%s$' % (anno))
        assert np.all(_[0] == chromos)
        for i in range(len(cpos)):
            assert np.all(_[1][i] == cpos[i])
        a = _[2].ravel()
        if a.sum() >= n_min:
            print('%s: %d' % (anno, a.sum()))
            e = evaluate(x[a], y[a], x_names, y_names)
            e['anno'] = anno
            es.append(e)

    es = pd.concat(es)
    es.sort_values('anno', inplace=True)
    return es


def eval_stats(x, y, x_names, y_names, chromos, cpos, stats_file, stats=None,
               nb_bin=5):
    if stats is None:
        f = h5.File(stats_file)
        stats = f[chromos[0]].keys()
        f.close()
        stats = list(filter(lambda x: x != 'pos', stats))
    es = []
    for stat in stats:
        s = []
        _ = io.read_stats(stats_file, chromos=chromos, pos=cpos,
                          regex='^%s$' % (stat))
        assert np.all(_[0] == chromos)
        for i in range(len(cpos)):
            assert np.all(_[1][i] == cpos[i])
        s = _[2].ravel()
        print('%s: %.3f' % (stat, s.mean()))
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
            e = evaluate(x[t], y[t], x_names, y_names)
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
            description='Correlate with conservation tracks')
        p.add_argument(
            'in_file',
            help='Input file')
        p.add_argument(
            'target_file',
            help='Target file')
        p.add_argument(
            '--dsets',
            help='Datasets to split',
            nargs='+')
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
            '--nb_sample',
            help='Number of samples',
            type=int)
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
            '--targets_file',
            help='Targets for computing correlations')
        p.add_argument(
            '--targets',
            help='Regex of targets',
            nargs='+',
            default=['(2i|ser)_w3000_(mean|var)'])
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
        x_pos = in_file['pos'].value
        x_chromo = in_file['chromo'].value
        x_chromos = [_.decode() for _ in np.unique(x_chromo)]
        dsets = opts.dsets
        if dsets is None:
            dsets = []
            for x in list(in_file.keys()):
                if x not in ['chromo', 'pos']:
                    dsets.append(x)

        target_file = h5.File(opts.target_file, 'r')
        data = dict(chromo=[], pos=[], x=[], y=[])
        for chromo in x_chromos:
            x = dict()
            _ = io.select_region(x_chromo, x_pos, chromo=chromo,
                                 nb_sample=opts.nb_sample)
            x['pos'] = x_pos[_]
            x['value'] = np.vstack([in_file[k][_] for k in dsets]).T
            y = dict()
            _ = target_file[chromo]
            y['pos'] = _['pos'].value
            y['value'] = _['value'].value.reshape(-1, 1)
            _ = np.in1d(x['pos'], y['pos'])
            for k in x.keys():
                x[k] = x[k][_]
            _ = np.in1d(y['pos'], x['pos'])
            for k in y.keys():
                y[k] = y[k][_]
            assert np.all(x['pos'] == y['pos'])
            data['chromo'].append(chromo)
            data['pos'].append(x['pos'])
            data['x'].append(x['value'])
            data['y'].append(y['value'])
            if opts.nb_sample is not None:
                break
        target_file.close()
        in_file.close()

        data['x'] = np.vstack(data['x']).astype('float32')
        data['y'] = np.vstack(data['y']).astype('float32')
        data['x_names'] = dsets
        data['y_names'] = ['cons']

        log.info('Evaluate global')
        print('Samples: %d' % (len(data['x'])))
        e = eval_global(data['x'], data['y'], data['x_names'], data['y_names'])
        if opts.verbose:
            print('Global statistics:')
            print(eval_to_str(e))
        if opts.out_dir is not None:
            write_output(e, 'global', opts.out_dir)
        if opts.sql_file is not None:
            ut.to_sql(opts.sql_file, e, 'global', sql_meta)

        if opts.annos_file is not None:
            log.info('Evaluate annos')
            e = eval_annos(data['x'], data['y'], data['x_names'],
                           data['y_names'], data['chromo'], data['pos'],
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
            e = eval_annos(data['x'], data['y'], data['x_names'],
                           data['y_names'], data['chromo'], data['pos'],
                           opts.stats_file, opts.stats, opts.stats_bins)
            if opts.verbose:
                print('Statistics:')
                print(eval_to_str(e))
            if opts.out_dir is not None:
                write_output(e, 'stats', opts.out_dir)
            if opts.sql_file is not None:
                ut.to_sql(opts.sql_file, e, 'stats', sql_meta)

        log.info('Done!')

        return 0

if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
