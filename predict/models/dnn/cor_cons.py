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
import predict.annos as pa


def read_bed_quant(path, chromos, cpos):
    d = pd.read_table(path, header=None, usecols=[0, 1, 2, 3])
    d.columns = ['chromo', 'start', 'end', 'value']
    d.chromo = d.chromo.str.lower().str.replace('chr', '')
    v = []
    for i, chromo in enumerate(chromos):
        dc = d.loc[d.chromo == chromo]
        if len(dc) == 0:
            v.append([])
        dc = dc.sort_values('start')
        pos = cpos[i]
        vc = np.empty(len(pos), 'float32')
        vc.fill(np.nan)
        idx = pa.in_which(pos, dc.start.values, dc.end.values)
        _ = idx >= 0
        vc[_] = dc.iloc[idx[_]]['value'].values
        v.append(vc)
    return v


def write_samples(path, act, z, filts, targets, group='global', nb_sample=None):
    idx = np.arange(len(act))
    if nb_sample is not None:
        np.random.shuffle(idx)
        idx = idx[:nb_sample]
    out_file = h5.File(path, 'a')
    g = out_file.create_group(group)
    g['act'] = act[idx]
    g['z'] = z[idx]
    g['targets'] = [x.encode() for x in targets]
    g['filts'] = filts
    out_file.close()


def evaluate(act, z, filts, targets):
    stats = []
    for f, filt in enumerate(filts):
        act_mean = act[:, f].mean()
        act_std = act[:, f].std()
        for t, target in enumerate(targets):
            s = []
            s.append(filt)
            s.append(target)
            zt = z[:, t]
            h = ~np.isnan(zt)
            zt = zt[h]
            at = act[h, f]
            s.append(len(zt))
            r = sps.pearsonr(at, zt)
            s.append(r[0])
            s.append(r[1])
            r = sps.spearmanr(at, zt)
            s.append(r[0])
            s.append(r[1])
            s.append(act_mean)
            s.append(act_std)
            stats.append(s)
    h = ['filt', 'target', 'n', 'rp', 'rp_pvalue', 'rs', 'rs_pvalue',
         'act_mean', 'act_std']
    stats = pd.DataFrame(stats, columns=h)
    return stats


def eval_global(act, z, filts, targets):
    return evaluate(act, z, filts, targets)


def eval_annos(act, z, filts, targets, chromos, cpos, annos_file,
               regexs=[r'loc_.+'], n_min=10, samples_file=None,
               samples_nb=None):
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
            e = evaluate(act[a], z[a], filts, targets)
            e['anno'] = anno
            es.append(e)
            if samples_file is not None:
                write_samples(samples_file, act[a], z[a], filts, targets,
                              group='annos/%s' % (anno), nb_sample=samples_nb)

    es = pd.concat(es)
    es.sort_values('anno', inplace=True)
    return es


def eval_stats(act, z, filts, targets, chromos, cpos, stats_file, stats=None,
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
            description='Correlate filter activations with conservation')
        p.add_argument(
            'filter_act_file',
            help='Filter activation file')
        p.add_argument(
            'cons_file',
            help='Conservation file')
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
            '--samples_file',
            help='Write samples to file')
        p.add_argument(
            '--samples_nb',
            help='Number of samples',
            type=int,
            default=1000)
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
        _ = ['chromo', 'pos', 'act']
        x_data = dict()
        for k in _:
            x_data[k] = in_file[k][:nb_sample]
        x_data = io.sort_cpos(x_data)
        chromos, cpos = io.cpos_to_list(x_data['chromo'], x_data['pos'])
        chromos = [x.decode() for x in chromos]

        if x_data['act'].ndim == 3:
            if opts.act_op == 'mean':
                f = np.mean
            else:
                f = np.max
            x_data['act'] = f(x_data['act'], axis=1)
        assert x_data['act'].ndim == 2
        x_data['value'] = x_data['act']

        nb_filt = x_data['act'].shape[1]
        filts = opts.filters
        if filts is None:
            filts = range(nb_filt)
        else:
            filts = ut.ranges_to_list(filts, 0, nb_filt - 1)
        nb_filt = len(filts)

        target_file = h5.File(opts.cons_file, 'r')
        off = 0
        data = dict(chromo=[], pos=[], x=[], y=[])
        for c, chromo in enumerate(chromos):
            x = {k: v[off: off + len(cpos[c])] for k, v in x_data.items()}
            assert np.all(x['chromo'] == chromo.encode())
            off += len(cpos[c])
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

        data['x'] = np.vstack(data['x']).astype('float32')
        data['y'] = np.vstack(data['y']).astype('float32')
        data['x_names'] = filts
        data['y_names'] = ['cons']

        target_file.close()
        in_file.close()

        log.info('Evaluate global')
        e = eval_global(data['x'], data['y'], data['x_names'], data['y_names'])
        if opts.verbose:
            print('Global statistics:')
            print(eval_to_str(e))
        if opts.out_dir is not None:
            write_output(e, 'global', opts.out_dir)
        if opts.sql_file is not None:
            ut.to_sql(opts.sql_file, e, 'global', sql_meta)
        if opts.samples_file is not None:
            write_samples(opts.samples_file, data['x'], data['y'],
                          data['x_names'], data['y_names'],
                          nb_sample=opts.samples_nb)

        if opts.annos_file is not None:
            log.info('Evaluate annos')
            e = eval_annos(data['x'], data['y'], data['x_names'],
                           data['y_names'], data['chromo'], data['pos'],
                           opts.annos_file, opts.annos,
                           samples_file=opts.samples_file,
                           samples_nb=opts.samples_nb)
            if opts.verbose:
                print('Annotation specific statistics:')
                print(eval_to_str(e))
            if opts.out_dir is not None:
                write_output(e, 'annos', opts.out_dir)
            if opts.sql_file is not None:
                ut.to_sql(opts.sql_file, e, 'annos', sql_meta)

        if opts.stats_file is not None:
            log.info('Evaluate stats')
            e = eval_stats(data['x'], data['y'], data['x_names'],
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
