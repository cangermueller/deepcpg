"""Feature extraction."""

from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import range


class KnnCpgFeatureExtractor(object):
    """Extract k CpG sites next to target sites. Exclude CpG sites at the
    same position.
    """

    def __init__(self, k=1):
        self.k = k

    def extract(self, x, y, ys):
        """Extract state and distance of k CpG sites next to target sites.
        Target site is excluded.

        Parameters
        ----------
        x: :class:`numpy.ndarray`
            :class:`numpy.ndarray` with target positions sorted in ascending
            order.
        y: :class:`numpy.ndarray`
            :class:`numpy.ndarray` with source positions sorted in ascending
            order.
        ys: :class:`numpy.ndarray`
            :class:`numpy.ndarray` with source CpG states.

        Returns
        -------
        tuple
            Tuple (cpg, dist) with numpy arrays of dimension (len(x), 2k):
            cpg: CpG states to the left (0:k) and right (k:2k)
            dist: Distances to the left (0:k) and right (k:2k)
        """

        n = len(x)
        m = len(y)
        k = self.k
        kk = 2 * self.k
        yc = self.__larger_equal(x, y)
        knn_cpg = np.empty((n, kk), dtype=np.float16)
        knn_cpg.fill(np.nan)
        knn_dist = np.empty((n, kk), dtype=np.float32)
        knn_dist.fill(np.nan)

        for i in range(n):
            # Left side
            yl = yc[i] - k
            yr = yc[i] - 1
            if yr >= 0:
                xl = 0
                xr = k - 1
                if yl < 0:
                    xl += np.abs(yl)
                    yl = 0
                xr += 1
                yr += 1
                knn_cpg[i, xl:xr] = ys[yl:yr]
                knn_dist[i, xl:xr] = np.abs(y[yl:yr] - x[i])

            # Right side
            yl = yc[i]
            if yl >= m:
                continue
            if x[i] == y[yl]:
                yl += 1
                if yl >= m:
                    continue
            yr = yl + k - 1
            xl = 0
            xr = k - 1
            if yr >= m:
                xr -= yr - m + 1
                yr = m - 1
            xl += k
            xr += k + 1
            yr += 1
            knn_cpg[i, xl:xr] = ys[yl:yr]
            knn_dist[i, xl:xr] = np.abs(y[yl:yr] - x[i])

        return (knn_cpg, knn_dist)

    def __larger_equal(self, x, y):
        """Return for each x[i] index j, s.t. y[j] >= x[i].

        Parameters
        ----------
        x: :class:`numpy.ndarray`
            :class:`numpy.ndarray` of with positions sorted in ascending order.
        y: :class:`numpy.ndarray`
            :class:`numpy.ndarray` of with positions sorted in ascending order.
        """

        n = len(x)
        m = len(y)
        rv = np.empty(n, dtype=np.int)
        i = 0
        j = 0
        while i < n and j < m:
            while j < m and x[i] > y[j]:
                j += 1
            rv[i] = j
            i += 1
        if i < n:
            # x[i] > y[m - 1]
            rv[i:] = m
        return rv


class IntervalFeatureExtractor(object):
    """Check if positions are in a list of intervals (start, end)."""

    @staticmethod
    def join_intervals(s, e):
        """Transform a list of possible overlapping intervals into
        non-overlapping intervals.

        Parameters
        ----------
        s: list
            List with start of interval sorted in ascending order.
        e: list
            List with end of interval.

        Returns
        -------
        tuple
            Tuple (s, e) of non-overlapping intervals.
        """

        rs = []
        re = []
        n = len(s)
        if n == 0:
            return (rs, re)
        l = s[0]
        r = e[0]
        for i in range(1, n):
            if s[i] > r:
                rs.append(l)
                re.append(r)
                l = s[i]
                r = e[i]
            else:
                r = max(r, e[i])
        rs.append(l)
        re.append(r)
        return (rs, re)

    @staticmethod
    def index_intervals(x, ys, ye):
        """Return for positions x[i] index j, s.t. ys[j] <= x[i] <= ye[j] or -1.
           Intervals must be non-overlapping!

        Parameters
        ----------
        x : list
            List of positions.
        ys: list
            List with start of interval sorted in ascending order.
        ye: list
            List with end of interval.

        Returns
        -------
        :class:`numpy.ndarray`
            :class:`numpy.ndarray` of same length than x with index or -1.
        """

        n = len(ys)
        m = len(x)
        rv = np.empty(m, dtype=np.int)
        rv.fill(-1)
        i = 0
        j = 0
        while i < n and j < m:
            while j < m and x[j] <= ye[i]:
                if x[j] >= ys[i]:
                    rv[j] = i
                j += 1
            i += 1
        return rv

    def extract(self, x, ys, ye):
        return self.index_intervals(x, ys, ye) >= 0


class KmersFeatureExtractor(object):

    def __init__(self, kmer_len, nb_char=4):
        """Extract kmer frequencies from integer sequences.

        Parameters
        ----------
        kmer_len: int
            Kmer length.
        nb_char: int
            Number of characters in alphabet.
        """
        self.kmer_len = kmer_len
        self.nb_char = nb_char
        self.nb_kmer = self.nb_char**self.kmer_len

    def __call__(self, seqs):
        """Extract kmer frequencies from integer sequences.

        Parameters
        ----------
        seqs: :class:`numpy.ndarray`
            :class:`numpy.ndarray` of size MxN, with M sequences of length N.

        Returns
        -------
        :class:`numpy.ndarray`
           :class:`numpy.ndarray` of size MxC with kmer frequencies.
        """
        nb_seq, seq_len = seqs.shape
        kmer_freq = np.zeros((nb_seq, self.nb_kmer), dtype=np.int32)
        vec = np.array([self.nb_char**i for i in range(self.kmer_len)],
                       dtype=np.int32)
        for i in range(nb_seq):
            for j in range(seq_len - self.kmer_len + 1):
                kmer = seqs[i, j:(j + self.kmer_len)]
                kmer_freq[i, kmer.dot(vec)] += 1
        return kmer_freq
