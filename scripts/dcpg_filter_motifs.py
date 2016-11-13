#!/usr/bin/env python

from collections import OrderedDict
import sys
import os
import os.path as pt

import argparse
import h5py as h5
import logging
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np
import subprocess
import pandas as pd

from deepcpg.utils import EPS, linear_weights
from deepcpg.data import dna

mpl.use('agg')
sns.set_style('darkgrid')

WEBLOGO_OPTS = '-X NO -Y NO --errorbars NO --fineprint ""'
WEBLOGO_OPTS += ' -C "#CB2026" A A'
WEBLOGO_OPTS += ' -C "#34459C" C C'
WEBLOGO_OPTS += ' -C "#FBB116" G G'
WEBLOGO_OPTS += ' -C "#0C8040" T T'

ALPHABET = dna.get_alphabet(False)
ALPHABET_R = OrderedDict([(value, key) for key, value in ALPHABET.items()])
MEME_ALPHABET = OrderedDict([('A', 0), ('C', 1), ('G', 2), ('T', 3)])


def zeropad_array(x, n, axis=0):
    pad_shape = list(x.shape)
    pad_shape[axis] += 2 * n
    pad = np.zeros(pad_shape, dtype=x.dtype)
    idx = [slice(0, x.shape[i]) for i in range(x.ndim)]
    idx[axis] = slice(n, n + x.shape[axis])
    pad[idx] = x
    return pad


def ranges_to_list(x, start=0, stop=None):
    s = set()
    for xi in x:
        xi = str(xi)
        if xi.find('-') >= 0:
            t = xi.split('-')
            if len(t) != 2:
                raise ValueError('Invalid range!')
            if len(t[0]) == 0:
                t[0] = start
            if len(t[1]) == 0:
                t[1] = stop
            s |= set(range(int(t[0]), int(t[1]) + 1))
        else:
            s.add(int(xi))
    s = sorted(list(s))
    return s


def format_out_of(out, of):
    return '%.1f%% (%d / %d)' % (out / of * 100, out, of)


def get_act_kmers(filter_act, filter_len, seqs, thr_per=0.5, thr_max=25000,
                  log=None):
    assert filter_act.shape[0] == seqs.shape[0]
    assert filter_act.shape[1] == seqs.shape[1]

    _thr_per = 0
    if thr_per:
        filter_act_mean = filter_act.mean()
        filter_act_norm = filter_act - filter_act_mean
        _thr_per = thr_per * filter_act_norm.max() + filter_act_mean
        if log:
            tmp = format_out_of(np.sum(filter_act >= _thr_per), filter_act.size)
            log('%s passed percentage threshold' % tmp)

    _thr_max = 0
    if thr_max:
        thr_max = min(thr_max, filter_act.size)
        _thr_max = np.percentile(filter_act,
                                 (1 - thr_max / filter_act.size) * 100)
        if log:
            tmp = format_out_of(np.sum(filter_act >= _thr_max), filter_act.size)
            log('%s passed maximum threshold' % tmp)

    kmers = []
    thr = max(_thr_per, _thr_max)
    idx = np.nonzero(filter_act >= thr)
    filter_del = filter_len // 2
    for k in range(len(idx[0])):
        i = int(idx[0][k])
        j = int(idx[1][k])
        if j < filter_del or j > (seqs.shape[1] - filter_len - 1):
            continue
        kmer = seqs[i, (j - filter_del):(j + filter_del + filter_len % 2)]
        kmers.append(kmer)
    kmers = np.array(kmers)

    return kmers


def write_kmers(kmers, filename):
    char_kmers = np.chararray(kmers.shape)
    for _char, _int in ALPHABET.items():
        char_kmers[kmers == _int] = _char

    with open(filename, 'w') as fh:
        for i, kmer in enumerate(char_kmers):
            print('>%d' % i, file=fh)
            print(kmer.tostring().decode(), file=fh)


def plot_filter_densities(densities, filename=None):
    sns.set(font_scale=1.3)
    fig, ax = plt.subplots()
    sns.distplot(densities, kde=False, ax=ax)
    ax.set_xlabel('Activation')
    if filename:
        fig.savefig(filename)
        plt.close()


def plot_filter_heatmap(weights, filename=None):
    param_range = abs(weights).max()

    fig, ax = plt.subplots(figsize=(weights.shape[1], weights.shape[0]))
    sns.heatmap(weights, cmap='RdYlBu_r', linewidths=0.2, vmin=-param_range,
                vmax=param_range, ax=ax)
    ax.set_xticklabels(range(1, weights.shape[1] + 1))
    labels = [ALPHABET_R[i] for i in reversed(range(weights.shape[0]))]
    ax.set_yticklabels(labels, rotation='horizontal', size=10)
    if filename:
        plt.savefig(filename)
        plt.close()


def plot_pca(act, pc_x=1, pc_y=2, labels=None, filename=None):
    act = act.T
    pca = PCA()
    pca.fit(act)
    eig_vec = pca.transform(act)
    data = pd.DataFrame(eig_vec)
    data.columns = ['PC%d' % i for i in range(data.shape[1])]
    data['act_mean'] = act.mean(axis=1)

    pc_x = 'PC%d' % pc_x
    pc_y = 'PC%d' % pc_y
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(data[pc_x], data[pc_y],
                         c=data['act_mean'], cmap='RdBu_r')
    ax.set_xlabel(pc_x)
    ax.set_ylabel(pc_y)
    fig.colorbar(scatter)
    if labels:
        for i, row in data.iterrows():
            ax.annotate('%d' % labels[i], xy=(row[pc_x], row[pc_y]),
                        fontsize=10)
    if filename:
        fig.savefig(filename)
        plt.close()


def map_alphabets(values, src_alphabet, dst_alphabet):
    assert len(src_alphabet) == len(dst_alphabet)
    _values = values.copy()
    for src_char, src_int in src_alphabet.items():
        _values[dst_alphabet[src_char]] = values[src_int]
    return _values


def open_meme(filename, seqs):
    nt_chars = list(ALPHABET.keys())

    nt_freq = []
    for nt_int in ALPHABET.values():
        nt_freq.append(np.sum(seqs == nt_int) + 1)
    nt_freq = np.array(nt_freq, dtype=np.float32)
    nt_freq = nt_freq / nt_freq.sum()
    nt_freq = map_alphabets(nt_freq, ALPHABET, MEME_ALPHABET)

    # open file for writing
    meme_file = open(filename, 'w')

    # print intro material
    print('MEME version 4', file=meme_file)
    print('', file=meme_file)
    print('ALPHABET= %s' % ''.join(list(MEME_ALPHABET.keys())), file=meme_file)
    print('', file=meme_file)
    print('Background letter frequencies:', file=meme_file)
    nt_freq_str = []
    for i, nt_char in enumerate(nt_chars):
        nt_freq_str.append('%s %.4f' % (nt_char, nt_freq[i]))
    print(' '.join(nt_freq_str), file=meme_file)
    print('', file=meme_file)

    return meme_file


def add_to_meme(meme_file, idx, pwm, nb_site, trim_thr=None):
    if trim_thr:
        start = 0
        while start < pwm.shape[0] and \
                info_content(pwm[start]) < trim_thr:
            start += 1

        end = len(pwm) - 1
        while end >= 0 and \
                info_content(pwm[end]) < trim_thr:
            end -= 1
        if start > end:
            return
        pwm = pwm[start:end]

    pwm = map_alphabets(pwm.T, ALPHABET, MEME_ALPHABET).T

    print('MOTIF filter%d' % idx, file=meme_file)
    tmp = 'letter-probability matrix: alength= %d w= %d nsites= %d'
    tmp = tmp % (len(MEME_ALPHABET), len(pwm), nb_site)
    print(tmp, file=meme_file)

    for row in pwm:
        row = ' '.join(['%.4f' % freq for freq in row])
        print(row, file=meme_file)
    print('', file=meme_file)


def get_pwm(seq_ali):
    # Initialize with 1 pseudocount
    pwm = np.ones((seq_ali.shape[1], len(ALPHABET)), dtype=np.float32)
    for i in range(pwm.shape[1]):
        pwm[:, i] = np.sum(seq_ali == i, axis=0)
    pwm = pwm / pwm.sum(axis=1).reshape(-1, 1)
    assert np.allclose(pwm.sum(axis=1), 1, atol=1e-4)
    return pwm


def info_content(pwm):
    pwm = np.atleast_2d(pwm)
    return np.sum(pwm * np.log2(pwm + EPS) + 0.5)


def read_tomtom(path):
    d = pd.read_table(path)
    d.rename(columns={'#Query ID': 'Query ID'}, inplace=True)
    d.columns = [x.lower() for x in d.columns]
    d['idx'] = [int(x) for x in d['query id'].str.replace('filter', '')]
    return d


def read_motif_proteins(meme_db_file):
    motif_proteins = {}
    for line in open(meme_db_file):
        a = line.split()
        if len(a) > 0 and a[0] == 'MOTIF':
            if a[2][0] == '(':
                motif_proteins[a[1]] = a[2][1:a[2].find(')')]
            else:
                motif_proteins[a[1]] = a[2]
    motif_proteins = {'target id': list(motif_proteins.keys()),
                      'protein': list(motif_proteins.values())}
    motif_proteins = pd.DataFrame(motif_proteins, columns=motif_proteins)
    return motif_proteins


def get_report(filter_stats_file, tomtom_file, motif_proteins):
    filter_stats = pd.read_table(filter_stats_file)
    tomtom = read_tomtom(tomtom_file)
    tomtom = tomtom.sort_values(['idx', 'q-value', 'e-value'])
    tomtom = tomtom.loc[:, ~tomtom.columns.isin(['query id', 'optimal offset'])]
    d = pd.merge(filter_stats, tomtom, on='idx', how='outer')
    d = pd.merge(d, motif_proteins, on='target id', how='left')
    d.index.name = None
    return d


def plot_logo(fasta_file, out_file, format='pdf', options=''):
    cmd = 'weblogo {opts} -s large < {inp} > {out} -F {f} 2> /dev/null'
    cmd = cmd.format(opts=options, inp=fasta_file, out=out_file, f=format)
    subprocess.call(cmd, shell=True)


def get_motif_from_weights(weights):
    idx = weights.argmax(axis=0)
    return ''.join([ALPHABET_R[i] for i in idx])


class App(object):

    def run(self, args):
        name = pt.basename(args[0])
        parser = self.create_parser(name)
        opts = parser.parse_args(args[1:])
        self.opts = opts
        return self.main(name, opts)

    def create_parser(self, name):
        p = argparse.ArgumentParser(
            prog=name,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Visualize filter motifs')
        p.add_argument(
            'in_file',
            help='Input file with filter activations and sequences')
        p.add_argument(
            '-o', '--out_dir',
            help='Output directory',
            default='.')
        p.add_argument(
            '-m', '--motif_db',
            help='MEME database for matching motifs')
        p.add_argument(
            '-a', '--act_thr_per',
            help='Percentage of maximum activation for selecting sites',
            default=0.5,
            type=float)
        p.add_argument(
            '--act_thr_max',
            help='Max number of sites to be selected',
            type=int,
            default=25000)
        p.add_argument(
            '--trim_thr',
            help='Threshold from trimming uninformative sites of PWM',
            type=float)
        p.add_argument(
            '--plot_dens',
            help='Plot filter activation densitities',
            action='store_true')
        p.add_argument(
            '--plot_heat',
            help='Plot filter heatmaps',
            action='store_true')
        p.add_argument(
            '--plot_pca',
            help='Plot activation principal components',
            action='store_true')
        p.add_argument(
            '--nb_sample_pca',
            help='Number of samples for PCA',
            type=int,
            default=1000)
        p.add_argument(
            '--fdr',
            help='FDR for motif matching',
            default=0.05,
            type=float)
        p.add_argument(
            '--nb_sample',
            help='Maximum # samples',
            type=int)
        p.add_argument(
            '--filters',
            help='Filters to be tested (starting from 0)',
            nargs='+')
        p.add_argument(
            '--WEBLOGO_OPTS',
            help='Weblogo options',
            default=WEBLOGO_OPTS)
        p.add_argument(
            '--weblogo_format',
            help='Weblogo plot format',
            default='pdf')
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

        log.info('Reading data')
        in_file = h5.File(opts.in_file, 'r')

        nb_sample = in_file['/act'].shape[0]
        if opts.nb_sample:
            nb_sample = min(opts.nb_sample, nb_sample)

        nb_filter = in_file['/act'].shape[-1]
        filters_idx = opts.filters
        if filters_idx is None:
            filters_idx = range(nb_filter)
        else:
            filters_idx = ranges_to_list(filters_idx, 0, nb_filter - 1)
            nb_filter = len(filters_idx)

        # Get only view on data to reduce memory usage. Possible since filters
        # can be processed independently.
        filters_act = in_file['/act']

        seqs = in_file['/inputs/dna'][:nb_sample]
        if seqs.shape[1] != filters_act.shape[1]:
            # Trim sequence length to length of activation layer
            tmp = (seqs.shape[1] - filters_act.shape[1]) // 2
            seqs = seqs[:, tmp:(tmp + filters_act.shape[1])]
            assert seqs.shape[1] == filters_act.shape[1]

        filters_weights = in_file['weights/weights']
        assert filters_weights.shape[1] == 1
        filters_weights = filters_weights[:, 0, :]
        filter_len = len(filters_weights)

        print('Filters: %d' % nb_filter)
        print('Filter len: %d' % filter_len)
        print('Samples: %d' % nb_sample)

        os.makedirs(opts.out_dir, exist_ok=True)
        sub_dirs = dict()
        names = ['logos', 'fa']
        if opts.plot_dens:
            names.append('dens')
        if opts.plot_heat:
            names.append('heat')
        for name in names:
            dirname = pt.join(opts.out_dir, name)
            sub_dirs[name] = dirname
            os.makedirs(dirname, exist_ok=True)

        meme_filename = pt.join(opts.out_dir, 'meme.txt')
        meme_file = open_meme(meme_filename, seqs)

        if opts.plot_pca:
            tmp = min(len(filters_act), opts.nb_sample_pca)
            log.info('Performing PCA on activations using %d samples' % tmp)
            # Down-sample activations to at most nb_sample_pca samples to reduce
            # memory usage and run-time.
            pca_act = filters_act[:tmp, :, filters_idx]

            act = pca_act.mean(axis=1)
            tmp = pt.join(opts.out_dir, 'pca_mean.pdf')
            plot_pca(act, labels=filters_idx, filename=tmp)

            weights = linear_weights(pca_act.shape[1])
            act = np.average(pca_act, 1, weights)
            tmp = pt.join(opts.out_dir, 'pca_wmean.pdf')
            plot_pca(act, labels=filters_idx, filename=tmp)

            act = pca_act.max(axis=1)
            tmp = pt.join(opts.out_dir, 'pca_max.pdf')
            plot_pca(act, labels=filters_idx, filename=tmp)

        log.info('Analyzing filters')
        log.info('-----------------')
        filter_stats = []
        for idx in filters_idx:
            log.info('Filter %d' % idx)
            filter_act = filters_act[:nb_sample, :, idx]
            filter_weights = filters_weights[:, :, idx].T
            assert len(filter_weights) == len(ALPHABET)

            stats = OrderedDict()
            stats['idx'] = idx
            stats['motif'] = get_motif_from_weights(filter_weights)
            stats['act_mean'] = filter_act.mean()
            stats['act_std'] = filter_act.std()
            stats['ic'] = 0
            stats['nb_site'] = 0
            stats = pd.Series(stats)
            filter_stats.append(stats)

            if stats['act_mean'] == 0:
                log.info('Dead filter -> skip')
                continue

            if opts.plot_dens:
                log.info('Plotting filter densities')
                tmp = pt.join(sub_dirs['dens'], '%03d.pdf' % idx)
                plot_filter_densities(np.ravel(filter_act), tmp)

            if opts.plot_heat:
                log.info('Plotting filter heatmap')
                tmp = pt.join(sub_dirs['heat'], '%03d.pdf' % idx)
                plot_filter_heatmap(filter_weights, tmp)

            log.info('Extracting activating kmers')
            act_kmers = get_act_kmers(filter_act, filter_len, seqs,
                                      thr_per=opts.act_thr_per,
                                      thr_max=opts.act_thr_max)
            stats.nb_site = len(act_kmers)

            if len(act_kmers) < 10:
                log.info('Only %d activating kmers -> skip' % len(act_kmers))
                continue

            log.info('Plotting sequence logo')
            logo_file = pt.join(sub_dirs['fa'], '%03d.fa' % idx)
            write_kmers(act_kmers, logo_file)
            plot_logo(logo_file, pt.join(sub_dirs['logos'], '%03d.pdf' % idx),
                      options=WEBLOGO_OPTS)

            log.info('Computing PWM')
            pwm = get_pwm(act_kmers)
            stats.ic = info_content(pwm)
            add_to_meme(meme_file, idx, pwm, len(act_kmers),
                        trim_thr=opts.trim_thr)

        meme_file.close()
        filter_stats = pd.DataFrame(filter_stats)
        for name in ['idx', 'nb_site']:
            filter_stats[name] = filter_stats[name].astype(np.int32)
        filter_stats.sort_values('act_mean', ascending=False, inplace=True)
        print()
        print('\nFilter statistics:')
        print(filter_stats.to_string())
        filter_stats.to_csv(pt.join(opts.out_dir, 'stats.csv'),
                            float_format='%.4f',
                            sep='\t', index=False)

        if opts.motif_db:
            log.info('Running tomtom')
            cmd = 'tomtom -dist pearson -thresh {thr} -oc {out_dir} ' + \
                '{meme_file} {motif_db}'
            cmd = cmd.format(thr=opts.fdr,
                             out_dir=pt.join(opts.out_dir, 'tomtom'),
                             meme_file=meme_filename,
                             motif_db=opts.motif_db)
            print('\n', cmd)
            subprocess.call(cmd, shell=True)

            motif_proteins = read_motif_proteins(opts.motif_db)
            tmp = pt.join(opts.out_dir, 'tomtom', 'motif_proteins.csv')
            motif_proteins.to_csv(tmp, sep='\t', index=False)

            report = get_report(
                pt.join(opts.out_dir, 'stats.csv'),
                pt.join(opts.out_dir, 'tomtom', 'tomtom.txt'),
                motif_proteins)
            report.sort_values(['idx', 'q-value', 'act_mean'],
                               ascending=[True, True, False], inplace=True)
            report.to_csv(pt.join(opts.out_dir, 'report.csv'), index=False,
                          sep='\t', float_format='%.3f')

            report_top = report.groupby('idx').first().reset_index()
            report_top.sort_values(['q-value', 'act_mean'],
                                   ascending=[True, False], inplace=True)
            report_top.to_csv(pt.join(opts.out_dir,
                                      'report_top.csv'), index=False,
                              sep='\t', float_format='%.3f')

            print('\nTomtom results:')
            print(report_top.to_string())

        in_file.close()
        log.info('Done!')
        return 0


if __name__ == '__main__':
    App().run(sys.argv)
