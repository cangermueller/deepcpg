#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import numpy as np
import pandas as pd
import progressbar
import h5py
import copy
import warnings

# sys.path.insert(0, pt.join(pt.dirname(pt.realpath(__file__)),
                           # '../../150221_biseq/biseq'))
import hdf
import data


class KmersExtractor(object):

    def __init__(self, k=1, chrs=['A', 'G', 'T', 'C']):
        self.k = k
        self.chrs = chrs
        self.b = len(chrs)
        self.ints = {c:i for i, c in enumerate(chrs)}
        self.vec = np.array([self.b**i for i in range(self.k)])

    def count(self):
        """Return total number of kmers."""
        return self.b**self.k

    def translate(self, s):
        """Translate str to int."""
        return np.array([self.ints[c] for c in s])

    def label(self, kmer):
        """Return label of kmer (int)."""
        label = ''
        for k in range(self.k):
            label += self.chrs[kmer % self.b]
            kmer = int(kmer / self.b)
        return label

    def labels(self, kmers=None):
        """Return labels of (all) kmers."""
        if kmers is None:
            kmers = range(self.count())
        return [self.label(kmer) for kmer in kmers]

    def kmers(self, s):
        """Extract kmers from s.

        Parameters
        ----------
        s: str sequence

        Returns
        -------
        kmers : kmers[i] is integer code of kmer at position i
        """
        st = self.translate(s)
        n = len(st)
        kmers = []
        for i in range(n - self.k + 1):
            kmers.append(st[i:i + self.k].dot(self.vec))
        return kmers

    def freq(self, s):
        """Return frequency of kmers in s.

        Parameters
        ----------
        s: str sequence

        Returns
        -------
        freq: numpy array of length self.count() with freq[i] as number of
                occurrences of kmer i in s.
        """
        f = np.zeros(self.count())
        for kmer in self.kmers(s):
            f[kmer] += 1
        return f


def adjust_pos(p, seq, target='CG'):
    for i in [0, -1, 1]:
        if seq[(p + i):(p + i + 2)] == target:
            return p + i
    return None


class Processor(object):

    def __init__(self, kext, delta):
        self.kext = kext
        self.delta = delta
        self.logger = None
        self.progbar = None
        self.seq_index = 1

    def log(self, msg):
        if self.logger is not None:
            self.logger(msg)

    def process_chromo(self, seq, pos):
        n = pos.shape[0]
        freq = np.zeros((n, self.kext.count()), dtype=np.int)
        seq = seq.upper()

        pb = None
        if self.progbar is not None:
            pb = copy.copy(self.progbar)
            pb.maxval = n

        num_adjusted = 0
        num_nocpg = 0
        num_stripped = 0
        num_inv = 0
        for i in range(n):
            if pb is not None:
                pb.update(i)
            p = pos[i] - self.seq_index
            if seq[p] not in self.kext.chrs:
                num_inv += 1
                continue
            q = adjust_pos(p, seq)
            if q is None:
                num_nocpg += 1
                q = p
            elif q != p:
                num_adjusted += 1
            p = q
            seq_win = seq[max(0, p - self.delta): min(len(seq), p + self.delta + 1)]
            t = len(seq_win)
            seq_win = seq_win.replace('N', '')
            if t != len(seq_win):
                num_stripped += 1
            freq[i] = self.kext.freq(seq_win)
        if pb is not None:
            pb.finish()
        self.log('%d (%.2f%%) positions adjusted to CpG.' % (num_adjusted, num_adjusted / n))
        self.log('%d (%.2f%%) no-CpG positions.' % (num_nocpg, num_nocpg / n))
        self.log('%d (%.2f%%) invalid positions.' % (num_inv, num_inv / n))
        self.log('%d (%.2f%%) of windows stripped.' % (num_stripped, num_stripped / n))

        freq = pd.DataFrame(freq, index=pos, columns=self.kext.labels())
        return freq


    def process(self, seq_path, pos, out_path):
        path, group = hdf.split_path(seq_path)
        seq_file = h5py.File(seq_path, 'r')
        seq_group = seq_file[group]
        out_path, out_group = hdf.split_path(out_path)
        for chromo in pos.chromo.unique():
            self.log('Chromosome %s ...' % (str(chromo)))
            p = pos.loc[pos.chromo == chromo].pos.values
            seq = seq_group[str(chromo)].value
            kmers = self.process_chromo(seq, p)
            t = data.format_chromo(chromo)
            kmers.to_hdf(out_path, pt.join(out_group, str(t)))
        seq_file.close()





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
            description='Extracts kmers from sequence at given positions')
        p.add_argument(
            'seq_file',
            help='HDF path to chromosome sequences')
        p.add_argument(
            'pos_file',
            help='Position file with pos and chromo column')
        p.add_argument(
            '-o', '--out_file',
            help='Output HDF path')
        p.add_argument(
            '-k', '--kmers',
            help='kmers length',
            type=int,
            default=2)
        p.add_argument(
            '--wlen',
            help='Length of sequence window at positions',
            default=100,
            type=int)
        p.add_argument(
            '--chromos',
            help='Only apply to these chromosome',
            nargs='+')
        p.add_argument(
            '--start',
            help='Start position on chromosome',
            type=int)
        p.add_argument(
            '--stop',
            help='Stop position on chromosome',
            type=int)
        p.add_argument(
            '--verbose', help='More detailed log messages', action='store_true')
        p.add_argument(
            '--log_file', help='Write log messages to file')
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

        log.info('Read positions ...')
        pos = pd.read_table(opts.pos_file, dtype={'chromo': str})
        if opts.chromos is not None:
            pos = pos.loc[pos.chromo.isin(opts.chromos)]
        if opts.start is not None:
            pos = pos.loc[pos.pos >= opts.start]
        if opts.stop is not None:
            pos = pos.loc[pos.pos <= opts.stop]
        pos.sort(['chromo', 'pos'], inplace=True)

        kext = KmersExtractor(opts.kmers)
        proc = Processor(kext, int(opts.wlen / 2))
        proc.progbar = progressbar.ProgressBar(term_width=80)
        proc.logger = lambda x: log.info(x)
        log.info('Process ...')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            proc.process(opts.seq_file, pos, opts.out_file)

        log.info('Done!')
        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
