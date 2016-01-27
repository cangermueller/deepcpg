#!/usr/bin/env python

import argparse
import sys
import logging
import os
import os.path as pt
import numpy as np
import h5py as h5
import subprocess
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

from predict.dna import int2char
from predict.models.dnn.utils import ArrayView
from predict import dna


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
            default='%s/data/motif_databases/CIS-BP/Mus_musculus.meme' % os.getenv('Pdata'))
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
        filter_outs = ArrayView(in_file['act'], stop=opts.nb_sample)

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

        # also save information contents
        filters_ic = []
        meme_out = meme_intro('%s/filters_meme.txt'%opts.out_dir, seqs)

        log.info('Create filter plots')
        for f in filters_list:
            log.info('Filter %d' % f)

            # plot filter parameters as a heatmap
            #  plot_filter_heat(filter_weights[f,:,:], '%s/filter%d_heat.pdf' % (opts.out_dir,f))

            # write possum motif file
            #  filter_possum(filter_weights[f,:,:], 'filter%d'%f, '%s/filter%d_possum.txt'%(opts.out_dir,f), opts.trim_filters)

            # plot weblogo of high scoring outputs
            import ipdb; ipdb.set_trace()
            plot_filter_logo(filter_outs[:, :, f], filter_size, seqs,
                             '%s/filter%d_logo'%(opts.out_dir,f),
                             maxpct_t=opts.act_t,
                             weblogo_opts=opts.weblogo_opts,
                             weblogo_format=opts.weblogo_format)

            # make a PWM for the filter
            filter_pwm, nsites = make_filter_pwm('%s/filter%d_logo.fa'%(opts.out_dir,f))

            if nsites < 10:
                # no information
                filters_ic.append(0)
            else:
                # compute and save information content
                filters_ic.append(info_content(filter_pwm))

                # add to the meme motif file
                meme_add(meme_out, f, filter_pwm, nsites, opts.trim_filters)

        meme_out.close()
        in_file.close()
        log.info('Done!')
        return 0


        #################################################################
        # annotate filters
        #################################################################
        # run tomtom
        log.info('Run tomtom')
        subprocess.call('tomtom -dist pearson -thresh 0.1 -oc %s/tomtom %s/filters_meme.txt %s' % (opts.out_dir, opts.out_dir, opts.meme_db), shell=True)

        # read in annotations
        filter_names = name_filters(num_filters, '%s/tomtom/tomtom.txt'%opts.out_dir, opts.meme_db)

        #################################################################
        # print a table of information
        #################################################################
        table_out = open('%s/table.txt'%opts.out_dir, 'w')

        # print header for later panda reading
        header_cols = ('', 'consensus', 'annotation', 'ic', 'mean', 'std')
        print('%3s  %19s  %10s  %5s  %6s  %6s' % header_cols, file=table_out)

        for f in range(num_filters):
            # collapse to a consensus motif
            consensus = filter_motif(filter_weights[f])

            # grab annotation
            annotation = '.'
            name_pieces = filter_names[f].split('_')
            if len(name_pieces) > 1:
                annotation = name_pieces[1]

            # plot density of filter output scores
            fmean, fstd = plot_score_density(np.ravel(filter_outs[:,f,:]), '%s/filter%d_dens.pdf' % (opts.out_dir,f))

            row_cols = (f, consensus, annotation, filters_ic[f], fmean, fstd)
            print('%-3d  %19s  %10s  %5.2f  %6.4f  %6.4f' % row_cols, file=table_out)

        table_out.close()


        #################################################################
        # global filter plots
        #################################################################
        # plot filter-sequence heatmap
        plot_filter_seq_heat(filter_outs, '%s/filter_seqs.pdf'%opts.out_dir)

        # plot filter-segment heatmap
        plot_filter_seg_heat(filter_outs, '%s/filter_segs.pdf'%opts.out_dir)
        plot_filter_seg_heat(filter_outs, '%s/filter_segs_raw.pdf'%opts.out_dir, whiten=False)

        # plot filter-target correlation heatmap
        #  plot_target_corr(filter_outs, seq_targets, filter_names, target_names, '%s/filter_target_cors_max.pdf'%opts.out_dir, 'max')

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

    nts = {'A':0, 'C':1, 'G':2, 'T':3}
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
        while ic_start < filter_pwm.shape[0] and info_content(filter_pwm[ic_start:ic_start+1]) < ic_t:
            ic_start += 1

        # trim PWM of uninformative suffix
        ic_end = filter_pwm.shape[0]-1
        while ic_end >= 0 and info_content(filter_pwm[ic_end:ic_end+1]) < ic_t:
            ic_end -= 1

    if ic_start < ic_end:
        print('MOTIF filter%d' % f, file=meme_out)
        print('letter-probability matrix: alength= 4 w= %d nsites= %d' % (ic_end-ic_start+1, nsites), file=meme_out)

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
    nts = {'A':0, 'C':1, 'G':2, 'T':3}

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


def name_filters(num_filters, tomtom_file, meme_db_file):
    ''' Name the filters using Tomtom matches.

    Attrs:
        num_filters (int) : total number of filters
        tomtom_file (str) : filename of Tomtom output table.
        meme_db_file (str) : filename of MEME db

    Returns:
        filter_names [str] :
    '''
    # name by number
    filter_names = ['f%d'%fi for fi in range(num_filters)]

    # name by protein
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

            filter_motifs.setdefault(fi,[]).append((qval,motif_id))

        tt_in.close()

        # assign filter's best match
        for fi in filter_motifs:
            top_motif = sorted(filter_motifs[fi])[0][1]
            filter_names[fi] += '_%s' % motif_protein[top_motif]

    return np.array(filter_names)



################################################################################
# plot_filter_seq_heat
#
# Plot a clustered heatmap of filter activations in
#
# Input
#  param_matrix: np.array of the filter's parameter matrix
#  out_pdf:
################################################################################
def plot_filter_seq_heat(filter_outs, out_pdf, whiten=True, drop_dead=True):
    # compute filter output means per sequence
    filter_seqs = filter_outs.mean(axis=2)

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

    hmin = np.percentile(filter_seqs[:,seqs_i], 0.1)
    hmax = np.percentile(filter_seqs[:,seqs_i], 99.9)

    sns.set(font_scale=0.3)

    plt.figure()
    sns.clustermap(filter_seqs[:,seqs_i], row_cluster=True, col_cluster=True, linewidths=0, xticklabels=False, vmin=hmin, vmax=hmax)
    plt.savefig(out_pdf)
    #out_png = out_pdf[:-2] + 'ng'
    #plt.savefig(out_png, dpi=300)
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
#  filter_outs
################################################################################
def plot_filter_seg_heat(filter_outs, out_pdf, whiten=True, drop_dead=True):
    b = filter_outs.shape[0]
    f = filter_outs.shape[1]
    l = filter_outs.shape[2]

    s = 5
    while l/float(s) - (l/s) > 0:
        s += 1
    print('%d segments of length %d' % (s,l/s))

    # split into multiple segments
    filter_outs_seg = np.reshape(filter_outs, (b, f, s, l/s))

    # mean across the segments
    filter_outs_mean = filter_outs_seg.max(axis=3)

    # break each segment into a new instance
    filter_seqs = np.reshape(np.swapaxes(filter_outs_mean, 2, 1), (s*b, f))

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

    hmin = np.percentile(filter_seqs[:,seqs_i], 0.1)
    hmax = np.percentile(filter_seqs[:,seqs_i], 99.9)

    sns.set(font_scale=0.3)
    if whiten:
        dist = 'euclidean'
    else:
        dist = 'cosine'

    plt.figure()
    sns.clustermap(filter_seqs[:,seqs_i], metric=dist, row_cluster=True, col_cluster=True, linewidths=0, xticklabels=False, vmin=hmin, vmax=hmax)
    plt.savefig(out_pdf)
    #out_png = out_pdf[:-2] + 'ng'
    #plt.savefig(out_png, dpi=300)
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
# filter_possum
#
# Write a Possum-style motif
#
# Input
#  param_matrix: np.array of the filter's parameter matrix
#  out_pdf:
################################################################################
def filter_possum(param_matrix, motif_id, possum_file, trim_filters=False, mult=200):
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
        print('MA %s' % ' '.join(['%.2f'%(mult*n) for n in param_matrix[:,ci]]), file=possum_out)
    print('END', file=possum_out)
    print('END', file=possum_out)

    possum_out.close()


################################################################################
# plot_filter_heat
#
# Plot a heatmap of the filter's parameters.
#
# Input
#  param_matrix: np.array of the filter's parameter matrix
#  out_pdf:
################################################################################
def plot_filter_heat(param_matrix, out_pdf):
    param_range = abs(param_matrix).max()

    plt.figure(figsize=(param_matrix.shape[1], 4))
    sns.heatmap(param_matrix, cmap='PRGn', linewidths=0.2, vmin=-param_range, vmax=param_range)
    ax = plt.gca()
    ax.set_xticklabels(range(1,param_matrix.shape[1]+1))
    ax.set_yticklabels('TGCA', rotation='horizontal', size=10)
    plt.savefig(out_pdf)
    plt.close()


################################################################################
# plot_filter_logo
#
# Plot a weblogo of the filter's occurrences
#
# Input
#  param_matrix: np.array of the filter's parameter matrix
#  out_pdf:
################################################################################
def plot_filter_logo(filter_outs, filter_size, seqs, out_prefix, raw_t=0, maxpct_t=None, weblogo_opts='', weblogo_format='pdf'):
    if maxpct_t:
        # TODO: max - min
        #  all_outs = np.ravel(filter_outs)
        all_outs = filter_outs
        all_outs_mean = all_outs.mean()
        all_outs_norm = all_outs - all_outs_mean
        raw_t = maxpct_t * all_outs_norm.max() + all_outs_mean

    # print fasta file of positive outputs
    filter_fasta_out = open('%s.fa' % out_prefix, 'w')
    filter_count = 0

    nb_sample = filter_outs.shape[0]
    seq_len = filter_outs.shape[1]
    filter_del = filter_size // 2
    for i in range(nb_sample):
        for j in range(filter_del, seq_len - filter_del):
            if filter_outs[i,j] > raw_t:
                kmer = seqs[i, (j - filter_del):(j + filter_del + filter_size % 2)]
                kmer = int2char(kmer)
                print('>%d_%d' % (i,j), file=filter_fasta_out)
                print(kmer, file=filter_fasta_out)
                filter_count += 1
    filter_fasta_out.close()

    # make weblogo
    if filter_count > 0:
        cmd = 'weblogo {opts} < {out}.fa > {out}.{f} -F {f}'
        cmd = cmd.format(opts=weblogo_opts,
                         out=out_prefix,
                         f=weblogo_format)
        subprocess.call(cmd, shell=True)


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
    sns.set(font_scale=1.3)
    plt.figure()
    sns.distplot(f_scores, kde=False)
    plt.xlabel('ReLU output')
    plt.savefig(out_pdf)
    plt.close()

    return f_scores.mean(), f_scores.std()


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


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    App().run(sys.argv)
