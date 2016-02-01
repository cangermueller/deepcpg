#!/usr/bin/env python

import argparse
import sys
import logging
import os
import os.path as pt
import numpy as np
import h5py as h5
import subprocess
from sklearn import preprocessing
import pandas as pd
import warnings
from scipy.stats import spearmanr

from predict.dna import int2char
from predict.models.dnn.utils import ArrayView
from predict import dna

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import seaborn as sns


weblogo_opts = '-X NO -Y NO --errorbars NO --fineprint ""'
weblogo_opts += ' -C "#CB2026" A A'
weblogo_opts += ' -C "#34459C" C C'
weblogo_opts += ' -C "#FBB116" G G'
weblogo_opts += ' -C "#0C8040" T T'


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
            help='Input file with filter activations and weights')
        p.add_argument(
            '-o', '--out_dir',
            help='Output directory',
            default='.')
        p.add_argument(
            '--nb_sample',
            help='Maximum # samples',
            type=int)
        p.add_argument(
            '--filters',
            help='Filters to be tested',
            nargs='+')
        p.add_argument(
            '--weblogo_opts',
            help='Weblogo options',
            default=weblogo_opts)
        p.add_argument(
            '--weblogo_format',
            help='Weblogo plot format',
            default='pdf')
        p.add_argument(
            '-a', '--act_t',
            help='Activation threshold (as proportion of max) to consider for PWM [Default: %default]',
            default=0.5,
            type=float)
        p.add_argument(
            '-m', '--meme_db',
            help='MEME database used to annotate motifs',
            default='%s/data/motif_databases/CIS-BP/Mus_musculus.meme' % os.getenv('Pr'))
        p.add_argument(
            '-t', '--trim_filters',
            help='Trim uninformative positions off the filter ends [Default: %default]',
            default=False,
            action='store_true')
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

        log.info('Read data')
        in_file = h5.File(opts.in_file, 'r')
        filter_weights = in_file['/filter/weights'].value.squeeze()
        assert len(filter_weights.shape) == 3
        assert filter_weights.shape[1] == 4
        seqs = ArrayView(in_file['s_x'], stop=opts.nb_sample)
        filter_act = ArrayView(in_file['act'], stop=opts.nb_sample)

        # store useful variables
        num_filters = filter_weights.shape[0]
        filter_size = filter_weights.shape[2]

        print('Filters: %d' % (num_filters))
        print('Filter len: %d' % (filter_size))
        print('Samples: %d' % (seqs.shape[0]))

        #################################################################
        # individual filter plots
        #################################################################
        filters_list = opts.filters
        if filters_list is None:
            filters_list = range(num_filters)
        else:
            filters_list = ranges_to_list(filters_list, 0, num_filters - 1)


        # plot filter-sequence heatmap
        log.info('Create filter sequence heatmap')
        fact = filter_act[:, :, filters_list]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            plot_filter_seq_heat(fact,
                                '%s/filter_seqs.pdf' % opts.out_dir)

        log.info('Create filter target heatmap')
        # plot correlation filter activation vs. predictions
        z = ArrayView(in_file['z'], stop=opts.nb_sample)
        target_names = [x.decode() for x in in_file['targets']]
        filter_names = [str(x) for x in filters_list]
        cor = corr_act_target(fact, z, filter_names, target_names, 'max')
        cor.to_csv(pt.join(opts.out_dir, 'filter_target_cor.csv'),
                   sep='\t', index=False)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            plot_corr_act_target(cor, pt.join(opts.out_dir,
                                            'filter_target_cors_max.pdf'))

        # also save information contents
        meme_out = meme_intro('%s/filters_meme.txt' % opts.out_dir, seqs)

        log.info('Analyze filters')
        stats = ['filt', 'motif', 'ic', 'acc_mean', 'acc_std']
        filter_stats = {x: [] for x in stats}
        for f in filters_list:
            log.info('Filter %d' % f)
            fweights = filter_weights[f, :, :]
            fact = filter_act[:, :, f]
            filter_stats['filt'].append(f)
            filter_stats['motif'].append(filter_motif(fweights))
            filter_stats['acc_mean'].append(fact.mean())
            filter_stats['acc_std'].append(fact.std())

            # plot weblogo of high scoring outputs
            path = '%s/filter%d_logo.fa' % (opts.out_dir, f)
            write_logo(fact, filter_size, seqs, path, maxpct_t=opts.act_t)
            plot_logo(path, options=weblogo_opts)

            # score density
            plot_score_density(np.ravel(fact),
                               '%s/filter%d_dens.pdf' % (opts.out_dir,f))

            # write possum motif file
            write_possum(fweights, 'filter%d' % f,
                         '%s/filter%d_possum.txt' % (opts.out_dir,f),
                         opts.trim_filters)

            # make a PWM for the filter
            t = '%s/filter%d_logo.fa' % (opts.out_dir, f)
            filter_pwm, nsites = make_filter_pwm(t)

            if nsites < 10:
                # no information
                filter_stats['ic'].append(0)
            else:
                # compute and save information content
                filter_stats['ic'].append(info_content(filter_pwm))

                # add to the meme motif file
                meme_add(meme_out, f, filter_pwm, nsites, opts.trim_filters)

        meme_out.close()
        filter_stats = pd.DataFrame(filter_stats, columns=stats)
        filter_stats.to_csv(pt.join(opts.out_dir, 'filter_stats.csv'),
                            sep='\t', index=False)

        # run tomtom
        log.info('Run tomtom')
        cmd = 'tomtom -dist pearson -thresh 0.1 -oc %s/tomtom ' + \
               '%s/filters_meme.txt %s 2> /dev/null'
        cmd = cmd % (opts.out_dir, opts.out_dir, opts.meme_db)
        subprocess.call(cmd, shell=True)

        # read in annotations
        summary = filter_summary(pt.join(opts.out_dir, 'filter_stats.csv'),
                                 pt.join(opts.out_dir, 'tomtom', 'tomtom.txt'),
                                 pt.join(opts.out_dir, opts.meme_db))
        summary.sort_values('acc_mean', ascending=False, inplace=True)
        summary.to_csv(pt.join(opts.out_dir, 'summary.csv'), index=False,
                       sep='\t', float_format='%.3f')

        in_file.close()
        log.info('Done!')
        return 0


def get_motif_proteins(meme_db_file):
    ''' Hash motif_id's to protein names using the MEME DB file '''
    motif_protein = {}
    for line in open(meme_db_file):
        a = line.split()
        if len(a) > 0 and a[0] == 'MOTIF':
            if a[2][0] == '(':
                motif_protein[a[1]] = a[2][1:a[2].find(')')]
            else:
                motif_protein[a[1]] = a[2]
    return motif_protein


def info_content(pwm, transpose=False):
    ''' Compute PWM information content '''
    pseudoc = 1e-9

    if transpose:
        pwm = np.transpose(pwm)

    ic = 0
    for i in range(pwm.shape[0]):
        for j in range(4):
            ic += 0.5 + pwm[i][j]*np.log2(pseudoc+pwm[i][j])

    return ic


def make_filter_pwm(filter_fasta):
    ''' Make a PWM for this filter from its top hits '''

    nts = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    pwm_counts = []
    nsites = 4 # pseudocounts
    for line in open(filter_fasta):
        if line[0] != '>':
            seq = line.rstrip()
            nsites += 1
            if len(pwm_counts) == 0:
                # initialize with the length
                for i in range(len(seq)):
                    pwm_counts.append(np.array([1.0]*4))

            # count
            for i in range(len(seq)):
                try:
                    pwm_counts[i][nts[seq[i]]] += 1
                except KeyError:
                    pwm_counts[i] += np.array([0.25]*4)

    # normalize
    pwm_freqs = []
    for i in range(len(pwm_counts)):
        pwm_freqs.append([pwm_counts[i][j]/float(nsites) for j in range(4)])

    return np.array(pwm_freqs), nsites-4


def meme_add(meme_out, f, filter_pwm, nsites, trim_filters=False):
    ''' Print a filter to the growing MEME file

    Attrs:
        meme_out : open file
        f (int) : filter index #
        filter_pwm (array) : filter PWM array
        nsites (int) : number of filter sites
    '''
    if not trim_filters:
        ic_start = 0
        ic_end = filter_pwm.shape[0]-1
    else:
        ic_t = 0.2

        # trim PWM of uninformative prefix
        ic_start = 0
        while ic_start < filter_pwm.shape[0] and \
                info_content(filter_pwm[ic_start:ic_start+1]) < ic_t:
            ic_start += 1

        # trim PWM of uninformative suffix
        ic_end = filter_pwm.shape[0]-1
        while ic_end >= 0 and info_content(filter_pwm[ic_end:ic_end+1]) < ic_t:
            ic_end -= 1

    if ic_start < ic_end:
        print('MOTIF filter%d' % f, file=meme_out)
        t = 'letter-probability matrix: alength= 4 w= %d nsites= %d'
        t = t % (ic_end-ic_start+1, nsites)
        print(t, file=meme_out)

        for i in range(ic_start, ic_end+1):
            print('%.4f %.4f %.4f %.4f' % tuple(filter_pwm[i]), file=meme_out)
        print('', file=meme_out)


def meme_intro(meme_file, seqs):
    ''' Open MEME motif format file and print intro

    Attrs:
        meme_file (str) : filename
        seqs [str] : list of strings for obtaining background freqs

    Returns:
        mem_out : open MEME file
    '''
    nts = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    # count
    nt_counts = [1]*4
    for i in range(len(seqs)):
        for nt in seqs[i]:
            try:
                nt_counts[nts[nt]] += 1
            except KeyError:
                pass

    # normalize
    nt_sum = float(sum(nt_counts))
    nt_freqs = [nt_counts[i]/nt_sum for i in range(4)]

    # open file for writing
    meme_out = open(meme_file, 'w')

    # print intro material
    print('MEME version 4', file=meme_out)
    print('', file=meme_out)
    print('ALPHABET= ACGT', file=meme_out)
    print('', file=meme_out)
    print('Background letter frequencies:', file=meme_out)
    print('A %.4f C %.4f G %.4f T %.4f' % tuple(nt_freqs), file=meme_out)
    print('', file=meme_out)

    return meme_out


def name_filters(filters_list, tomtom_file, meme_db_file):
    ''' Name the filters using Tomtom matches.

    Attrs:
        num_filters (int) : total number of filters
        tomtom_file (str) : filename of Tomtom output table.
        meme_db_file (str) : filename of MEME db

    Returns:
        filter_names [str] :
    '''
    # name by number
    filter_names = {fi: 'f%d' % fi for fi in filters_list}

    if tomtom_file is not None and meme_db_file is not None:
        motif_protein = get_motif_proteins(meme_db_file)

        # hash motifs and q-value's by filter
        filter_motifs = {}

        tt_in = open(tomtom_file)
        tt_in.readline()
        for line in tt_in:
            a = line.split()
            fi = int(a[0][6:])
            motif_id = a[1]
            qval = float(a[5])

            filter_motifs.setdefault(fi, []).append((qval, motif_id))

        tt_in.close()

        # assign filter's best match
        for fi in filter_motifs:
            top_motif = sorted(filter_motifs[fi])[0][1]
            filter_names[fi] += '_%s' % motif_protein[top_motif]

    return filter_names


################################################################################
# plot_filter_seq_heat
#
# Plot a clustered heatmap of filter activations in
#
# Input
#  param_matrix: np.array of the filter's parameter matrix
#  out_pdf:
################################################################################
def plot_filter_seq_heat(filter_act, out_pdf, whiten=True, drop_dead=True):
    # mean activation seq x filter
    filter_seqs = filter_act.mean(axis=1)

    # whiten
    if whiten:
        filter_seqs = preprocessing.scale(filter_seqs)

    # transpose -> filter x seq
    filter_seqs = np.transpose(filter_seqs)

    if drop_dead:
        filter_stds = filter_seqs.std(axis=1)
        filter_seqs = filter_seqs[filter_stds > 0]

    # downsample sequences
    seqs_i = np.random.randint(0, filter_seqs.shape[1], 500)

    hmin = np.percentile(filter_seqs[:, seqs_i], 0.1)
    hmax = np.percentile(filter_seqs[:, seqs_i], 99.9)

    sns.set(font_scale=0.3)

    plt.figure()
    sns.clustermap(filter_seqs[:,seqs_i], row_cluster=True, col_cluster=True,
                   linewidths=0, xticklabels=False, vmin=hmin, vmax=hmax)
    plt.savefig(out_pdf)
    plt.close()


################################################################################
# plot_filter_seq_heat
#
# Plot a clustered heatmap of filter activations in sequence segments.
#
# Mean doesn't work well for the smaller segments for some reason, but taking
# the max looks OK. Still, similar motifs don't cluster quite as well as you
# might expect.
#
# Input
#  filter_act
################################################################################
def plot_filter_seg_heat(filter_act, out_pdf, whiten=True, drop_dead=True):
    filter_act = np.swapaxes(filter_act, 1, 2)
    b = filter_act.shape[0]
    f = filter_act.shape[1]
    l = filter_act.shape[2]

    s = 5
    while l/float(s) - (l/s) > 0:
        s += 1
    print('%d segments of length %d' % (s, l / s))

    # split into multiple segments
    filter_act_seg = np.reshape(filter_act, (b, f, s, l/s))

    # mean across the segments
    filter_act_mean = filter_act_seg.max(axis=3)

    # break each segment into a new instance
    filter_seqs = np.reshape(np.swapaxes(filter_act_mean, 2, 1), (s*b, f))

    # whiten
    if whiten:
        filter_seqs = preprocessing.scale(filter_seqs)

    # transpose
    filter_seqs = np.transpose(filter_seqs)

    if drop_dead:
        filter_stds = filter_seqs.std(axis=1)
        filter_seqs = filter_seqs[filter_stds > 0]

    # downsample sequences
    seqs_i = np.random.randint(0, filter_seqs.shape[1], 500)

    hmin = np.percentile(filter_seqs[:, seqs_i], 0.1)
    hmax = np.percentile(filter_seqs[:, seqs_i], 99.9)

    sns.set(font_scale=0.3)
    if whiten:
        dist = 'euclidean'
    else:
        dist = 'cosine'

    plt.figure()
    sns.clustermap(filter_seqs[:,seqs_i], metric=dist, row_cluster=True,
                   lcol_cluster=True, linewidths=0, xticklabels=False,
                   vmin=hmin, vmax=hmax)
    plt.savefig(out_pdf)
    plt.close()


################################################################################
# filter_motif
#
# Collapse the filter parameter matrix to a single DNA motif.
#
# Input
#  param_matrix: np.array of the filter's parameter matrix
#  out_pdf:
################################################################################
def filter_motif(param_matrix):
    return dna.int2char(param_matrix.argmax(axis=0))


################################################################################
# write_possum
#
# Write a Possum-style motif
#
# Input
#  param_matrix: np.array of the filter's parameter matrix
#  out_pdf:
################################################################################
def write_possum(param_matrix, motif_id, possum_file, trim_filters=False,
                 mult=200):
    # possible trim
    trim_start = 0
    trim_end = param_matrix.shape[1]-1
    trim_t = 0.3
    if trim_filters:
        # trim PWM of uninformative prefix
        while np.max(param_matrix[:,trim_start]) - np.min(param_matrix[:,trim_start]) < trim_t:
            trim_start += 1

        # trim PWM of uninformative suffix
        while np.max(param_matrix[:,trim_end]) - np.min(param_matrix[:,trim_end]) < trim_t:
            trim_end -= 1

    possum_out = open(possum_file, 'w')
    print('BEGIN GROUP', file=possum_out)
    print('BEGIN FLOAT', file=possum_out)
    print('ID %s' % motif_id, file=possum_out)
    print('AP DNA', file=possum_out)
    print('LE %d' % (trim_end+1-trim_start), file=possum_out)
    for ci in range(trim_start,trim_end+1):
        print('MA %s' % ' '.join(['%.2f'% (mult * n) for n in param_matrix[:, ci]]), file=possum_out)
    print('END', file=possum_out)
    print('END', file=possum_out)

    possum_out.close()


################################################################################
#
# plot_filter_heat
# Plot a heatmap of the filter's parameters.
#
# Input
#  param_matrix: np.array of the filter's parameter matrix
#  out_pdf:
################################################################################
def plot_filter_heat(param_matrix, out_pdf):
    param_range = abs(param_matrix).max()

    plt.figure(figsize=(param_matrix.shape[1], 4))
    sns.heatmap(param_matrix, cmap='PRGn', linewidths=0.2, vmin=-param_range,
                vmax=param_range)
    ax = plt.gca()
    ax.set_xticklabels(range(1,param_matrix.shape[1]+1))
    labels = dna.int2char(range(param_matrix.shape[0]))
    ax.set_yticklabels(labels, rotation='horizontal', size=10)
    plt.savefig(out_pdf)
    plt.close()


def plot_logo(fasta_file, out_file=None, format='pdf', options=''):
    if out_file is None:
        out_file = '%s.%s' % (pt.splitext(fasta_file)[0], format)
    cmd = 'weblogo {opts} -s large < {inp} > {out} -F {f} 2> /dev/null'
    cmd = cmd.format(opts=options, inp=fasta_file, out=out_file, f=format)
    subprocess.call(cmd, shell=True)


def write_logo(filter_act, filter_size, seqs, out_file, maxpct_t=None):
    raw_t = 0
    if maxpct_t:
        raw_t = filter_act.min() + maxpct_t * (filter_act.max() - filter_act.min())

    # print fasta file of positive outputs
    filter_fasta_out = open(out_file, 'w')
    filter_count = 0

    nb_sample = filter_act.shape[0]
    seq_len = filter_act.shape[1]
    filter_del = filter_size // 2
    for i in range(nb_sample):
        for j in range(filter_del, seq_len - filter_del):
            if filter_act[i,j] > raw_t:
                kmer = seqs[i, (j - filter_del):(j + filter_del + filter_size % 2)]
                kmer = int2char(kmer)
                print('>%d_%d' % (i, j), file=filter_fasta_out)
                print(kmer, file=filter_fasta_out)
                filter_count += 1
    filter_fasta_out.close()


################################################################################
# plot_score_density
#
# Plot the score density and print to the stats table.
#
# Input
#  param_matrix: np.array of the filter's parameter matrix
#  out_pdf:
################################################################################
def plot_score_density(f_scores, out_pdf):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sns.set(font_scale=1.3)
        plt.figure()
        sns.distplot(f_scores, kde=False)
        plt.xlabel('ReLU output')
        plt.savefig(out_pdf)
        plt.close()


################################################################################
# plot_target_corr
#
# Plot a clustered heatmap of correlations between filter activations and
# targets.
#
# Input
#  filter_outs:
#  filter_names:
#  target_names:
#  out_pdf:
################################################################################
def corr_act_target(filter_act, seq_targets, filter_names, target_names,
                    seq_op='mean'):
    num_seqs = filter_act.shape[0]
    num_targets = len(target_names)

    if seq_op == 'mean':
        filter_act_seq = filter_act.mean(axis=1)
    else:
        filter_act_seq = filter_act.max(axis=1)

    # std is sequence by filter.
    filter_seqs_std = filter_act_seq.std(axis=0)
    filter_act_seq = filter_act_seq[:, filter_seqs_std > 0]
    filter_names_live = [filter_names[x] for x in np.where(filter_seqs_std > 0)[0]]

    filter_target_cors = np.zeros((len(filter_names_live), num_targets))
    for fi in range(len(filter_names_live)):
        for ti in range(num_targets):
            cor, p = spearmanr(filter_act_seq[:, fi],
                               seq_targets[:num_seqs, ti])
            filter_target_cors[fi, ti] = cor

    cor_df = pd.DataFrame(filter_target_cors, index=filter_names_live,
                          columns=target_names)
    return cor_df


def plot_corr_act_target(d, out_pdf):
    sns.set(font_scale=0.3)
    plt.figure()
    sns.clustermap(d, cmap='BrBG', center=0, figsize=(8, 10))
    plt.savefig(out_pdf)
    plt.close()


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


def read_tomtom(path):
    d = pd.read_table(path)
    d.rename(columns={'#Query ID': 'Query ID'}, inplace=True)
    d.columns = [x.lower() for x in d.columns]
    d['filt'] = [int(x) for x in d['query id'].str.replace('filter', '')]
    return d


def filter_summary(filter_stats_file, tomtom_file, meme_db_file):
    filter_stats = pd.read_table(filter_stats_file)
    tomtom = read_tomtom(tomtom_file)
    tomtom = tomtom.sort_values(['filt', 'q-value', 'e-value'])
    tomtom = tomtom.groupby('filt').first().reset_index()
    tomtom = tomtom.loc[:, ~tomtom.columns.isin(['query id', 'optimal offset'])]
    motif_protein = get_motif_proteins(meme_db_file)
    motif_protein = {'target id': list(motif_protein.keys()),
                     'protein': list(motif_protein.values())}
    motif_protein = pd.DataFrame(motif_protein, columns=motif_protein)
    d = pd.merge(filter_stats, tomtom, on='filt', how='outer')
    d = pd.merge(d, motif_protein, on='target id', how='left')
    cols = list(filter_stats.columns)
    cols.extend(['target id', 'protein', 'e-value', 'q-value', 'overlap',
                 'query consensus', 'target consensus', 'orientation'])
    d = d.loc[:, cols]
    return d


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    App().run(sys.argv)
