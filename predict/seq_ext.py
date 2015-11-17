import os.path as pt
import numpy as np
import pandas as pd
import h5py
import copy

from predict import hdf


def adjust_pos(p, seq, target='CG'):
    for i in [0, -1, 1]:
        if seq[(p + i):(p + i + 2)] == target:
            return p + i
    return None


class Processor(object):

    def __init__(self, wlen):
        self.delta = wlen // 2
        self.wlen = self.delta * 2 + 1
        self.logger = None
        self.progbar = None
        self.seq_index = 1
        self.trans = {'A': 0, 'G': 1, 'T': 2, 'C': 3, 'N': 4}

    def log(self, msg):
        if self.logger is not None:
            self.logger(msg)

    def encode(self, s):
        return [self.trans[c] for c in s]

    def process_chromo(self, seq, pos):
        n = pos.shape[0]
        seq = seq.upper()

        seqs = np.zeros((n, self.wlen), dtype=np.int8)

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
            q = adjust_pos(p, seq)
            if q is None:
                num_inv += 1
            elif p != q:
                num_adjusted += 1
            seq_win = seq[max(0, p - self.delta): min(len(seq), p + self.delta + 1)]
            if p != q:
                print(seq_win)
                continue
            assert len(seq_win) == self.wlen
            seqs[i] = self.encode(seq_win)


        if pb is not None:
            pb.finish()
        self.log('%d (%.2f%%) positions adjusted to CpG.' % (num_adjusted, num_adjusted / n))
        self.log('%d (%.2f%%) no-CpG positions.' % (num_nocpg, num_nocpg / n))
        self.log('%d (%.2f%%) invalid positions.' % (num_inv, num_inv / n))
        self.log('%d (%.2f%%) of windows stripped.' % (num_stripped, num_stripped / n))
        return seqs


    def process(self, seq_path, pos, out_path):
        path, group = hdf.split_path(seq_path)
        seq_file = h5py.File(seq_path, 'r')
        seq_group = seq_file[group]
        out_path, out_group = hdf.split_path(out_path)
        for chromo in pos.chromo.unique():
            self.log('Chromosome %s ...' % (str(chromo)))
            p = pos.loc[pos.chromo == chromo].pos.values
            seq = seq_group[str(chromo)].value
            seqs = self.process_chromo(seq, p)
            seqs = pd.DataFrame(seqs, index=p)
            seqs.to_hdf(out_path, pt.join(out_group, str(chromo)))
        seq_file.close()
