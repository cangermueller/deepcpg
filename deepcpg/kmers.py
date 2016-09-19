import numpy as np
import pandas as pd
import h5py

from predict import hdf


class KmersExtractor(object):

    def __init__(self, k=1, chrs=['A', 'G', 'T', 'C']):
        self.k = k
        self.chrs = chrs
        self.b = len(chrs)
        self.ints = {c: i for i, c in enumerate(chrs)}
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
        self.seq_index = 1

    def log(self, msg):
        if self.logger is not None:
            self.logger(msg)

    def process_chromo(self, seq, pos):
        n = pos.shape[0]
        freq = np.zeros((n, self.kext.count()), dtype=np.int)
        seq = seq.upper()

        num_adjusted = 0
        num_nocpg = 0
        num_stripped = 0
        num_inv = 0
        for i in range(n):
            if self.logger is not None:
                if i == 0 or i == n - 1 or i % (n // 100) == 0:
                    self.logger('%6.1f (%d / %d)' % ((i + 1) / n * 100, i + 1, n))
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
        self.log('%d (%.2f%%) positions adjusted to CpG.' % (num_adjusted, num_adjusted / n))
        self.log('%d (%.2f%%) no-CpG positions.' % (num_nocpg, num_nocpg / n))
        self.log('%d (%.2f%%) invalid positions.' % (num_inv, num_inv / n))
        self.log('%d (%.2f%%) of windows stripped.' % (num_stripped, num_stripped / n))

        return freq
        freq = pd.DataFrame(freq, index=pos, columns=self.kext.labels())
        return freq

    def process(self, seq_path, pos, out_path):
        path, group = hdf.split_path(seq_path)
        seq_file = h5py.File(seq_path, 'r')
        seq_group = seq_file[group]
        out_path, out_group = hdf.split_path(out_path)
        out_file = h5py.File(out_path, 'a')
        if out_group not in out_file:
            out_file.create_group(out_group)
        g = out_file[out_group]
        for chromo in pos.chromo.unique():
            self.log('Chromosome %s ...' % (str(chromo)))
            p = pos.loc[pos.chromo == chromo].pos.values
            seq = seq_group[str(chromo)].value
            kmers = self.process_chromo(seq, p)
            if chromo in g:
                del g[chromo]
            gc = g.create_group(chromo)
            gc['pos'] = p
            gc.create_dataset('kmers', data=kmers, compression='gzip')
            gc['labels'] = [x.encode() for x in self.kext.labels()]
        out_file.close()
        seq_file.close()
