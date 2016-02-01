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
            description='Visualize CpG filter motifs')
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
        assert len(filter_weights.shape) == 4
        assert filter_weights.shape[1] == 2
        seqs = ArrayView(in_file['c_x'], stop=opts.nb_sample)
        filter_act = ArrayView(in_file['act'], stop=opts.nb_sample)

        nb_filter = filter_weights.shape[0]
        filter_len = filter_weights.shape[3]

        print('Filters: %d' % (nb_filter))
        print('Filter len: %d' % (filter_len))
        print('Samples: %d' % (seqs.shape[0]))




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
            write_logo(fact, filter_len, seqs, path, maxpct_t=opts.act_t)
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


def plot_logo(fasta_file, out_file=None, format='pdf', options=''):
    if out_file is None:
        out_file = '%s.%s' % (pt.splitext(fasta_file)[0], format)
    cmd = 'weblogo {opts} -s large < {inp} > {out} -F {f} 2> /dev/null'
    cmd = cmd.format(opts=options, inp=fasta_file, out=out_file, f=format)
    subprocess.call(cmd, shell=True)



if __name__ == '__main__':
    App().run(sys.argv)
