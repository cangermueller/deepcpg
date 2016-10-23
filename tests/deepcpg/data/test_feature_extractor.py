import numpy as np
import numpy.testing as npt

from deepcpg.data import feature_extractor as fe
from deepcpg.data import dna


class TestKnnCpgFeatureExtractor(object):

    def test_larger_equal(self):
        # y: 1 5 8 15
        e = fe.KnnCpgFeatureExtractor()
        f = e._KnnCpgFeatureExtractor__larger_equal
        y = np.array([1, 5, 8, 15])
        x = np.array([4, 6, 10])

        expect = np.array([1, 2, 3])
        result = f(x, y)
        npt.assert_array_equal(result, expect)

        x = np.array([-1, 0, 5, 14, 15, 16, 20])
        expect = np.array([0, 0, 1, 3, 3, 4, 4])
        result = f(x, y)
        npt.assert_array_equal(result, expect)

    def _compare(self, results, expected):
        ctr = expected.shape[1] // 2
        npt.assert_array_almost_equal(results[0], expected[:, :ctr], 4,
                                      'States mismatch!')
        npt.assert_array_almost_equal(results[1], expected[:, ctr:], 4,
                                      'Distances mismatch!')

    def test_extract_k1(self):
        y = np.array([1, 3, 5, 8, 15])
        ys = np.array([0, 0, 1, 1, 0])

        x = np.array([2, 6, 10])
        expect = np.array([[0, 0, 1, 1],
                           [1, 1, 1, 2],
                           [1, 0, 2, 5]])
        result = fe.KnnCpgFeatureExtractor(1).extract(x, y, ys)
        self._compare(result, expect)

        x = np.array([0, 1, 3, 11, 15, 20])
        expect = np.array([[np.nan, 0, np.nan, 1],
                           [np.nan, 0, np.nan, 2],
                           [0, 1, 2, 2],
                           [1, 0, 3, 4],
                           [1, np.nan, 7, np.nan],
                           [0, np.nan, 5, np.nan]])
        result = fe.KnnCpgFeatureExtractor(1).extract(x, y, ys)
        self._compare(result, expect)

    def test_extract_k2(self):
        y = np.array([1, 3, 5, 8, 15])
        ys = np.array([0, 0, 1, 1, 0])
        x = np.array([2, 6, 10])
        expect = np.array([[np.nan, 0, 0, 1, np.nan, 1, 1, 3],
                           [0, 1, 1, 0, 3, 1, 2, 9],
                           [1, 1, 0, np.nan, 5, 2, 5, np.nan]])
        result = fe.KnnCpgFeatureExtractor(2).extract(x, y, ys)
        self._compare(result, expect)

        x = np.array([0, 1, 3, 8, 11, 15, 20])
        expect = np.array([[np.nan, np.nan, 0, 0, np.nan, np.nan, 1, 3],
                           [np.nan, np.nan, 0, 1, np.nan, np.nan, 2, 4],
                           [np.nan, 0, 1, 1, np.nan, 2, 2, 5],
                           [0, 1, 0, np.nan, 5, 3, 7, np.nan],
                           [1, 1, 0, np.nan, 6, 3, 4, np.nan],
                           [1, 1, np.nan, np.nan, 10, 7, np.nan, np.nan],
                           [1, 0, np.nan, np.nan, 12, 5, np.nan, np.nan]])
        result = fe.KnnCpgFeatureExtractor(2).extract(x, y, ys)
        self._compare(result, expect)

    def test_extract_k3(self):
        y = np.array([1, 3, 5, 8, 15])
        ys = np.array([0, 0, 1, 1, 0])
        x = np.array([2, 3, 6, 10, 15])
        expect = np.array([[np.nan, np.nan, 0, 0, 1, 1, np.nan, np.nan, 1, 1, 3, 6],
                           [np.nan, np.nan, 0, 1, 1, 0, np.nan, np.nan, 2, 2, 5, 12],
                           [0, 0, 1, 1, 0, np.nan, 5, 3, 1, 2, 9, np.nan],
                           [0, 1, 1, 0, np.nan, np.nan, 7, 5, 2, 5, np.nan, np.nan],
                           [0, 1, 1, np.nan, np.nan, np.nan, 12, 10, 7, np.nan, np.nan, np.nan]])
        result = fe.KnnCpgFeatureExtractor(3).extract(x, y, ys)
        self._compare(result, expect)


class TestIntervalFeatureExtractor(object):

    def test_join_intervals(self):
        f = fe.IntervalFeatureExtractor.join_intervals

        s, e = f([], [])
        assert len(s) == 0
        assert len(e) == 0

        s = [1, 3, 6]
        e = [2, 4, 10]
        expect = (s, e)
        result = f(s, e)
        assert result == expect

        x = np.array([[1, 2],
                      [3, 4], [4, 5],
                      [6, 8], [8, 8], [8, 9],
                      [10, 15], [10, 11], [11, 14], [14, 16]])
        expect = [[1, 2], [3, 5], [6, 9], [10, 16]]
        result = np.array(f(x[:, 0], x[:, 1])).T
        npt.assert_array_equal(result, expect)

    def test_index_intervals(self):
        f = fe.IntervalFeatureExtractor.index_intervals
        ys = [2, 4, 12, 17]
        ye = [2, 8, 15, 18]

        x = []
        expect = []
        result = f(x, ys, ye)
        npt.assert_array_equal(result, expect)

        x = [-1, 3, 9, 19]
        expect = [-1, -1, -1, -1]
        result = f(x, ys, ye)
        npt.assert_array_equal(result, expect)

        x = [-1, 2, 2, 3, 4, 8, 15, 16]
        expect = [-1, 0, 0, -1, 1, 1, 2, -1]
        result = f(x, ys, ye)
        npt.assert_array_equal(result, expect)

    def test_extract(self):
        ys = [2, 4, 12, 17]
        ye = [2, 8, 15, 18]
        e = fe.IntervalFeatureExtractor()

        x = [-1, 2, 2, 3, 4, 8, 15, 16]
        expect = [False, True, True, False, True, True, True, False]
        result = e.extract(x, ys, ye)
        npt.assert_array_equal(result, expect)


class TestKmersFeatureExtractor(object):

    def _translate_seqs(self, seqs):
        if not isinstance(seqs, list):
            seqs = [seqs]
        _seqs = np.array([dna.char2int(seq) for seq in seqs], dtype=np.int32)
        return _seqs

    def _kmer_idx(self, kmer):
        idx = 0
        for i, c in enumerate(kmer):
            idx += dna.CHAR_TO_INT[c] * 4**i
        return idx

    def _freq(self, kmer_freq):
        kmer_len = 4**len(list(kmer_freq.keys())[0])
        _freq = np.zeros(kmer_len)
        for kmer, freq in kmer_freq.items():
            _freq[self._kmer_idx(kmer)] = freq
        return _freq

    def test_k1(self):
        ext = fe.KmersFeatureExtractor(1)

        seqs = self._translate_seqs('AGGTTCCC')
        expect = self._freq({'A': 1, 'G': 2, 'T': 2, 'C': 3})
        expect = np.array([expect])
        actual = ext(seqs)
        npt.assert_array_equal(actual, expect)

        seqs = self._translate_seqs('AGTGGGTTCCC')
        expect = self._freq({'A': 1, 'G': 4, 'T': 3, 'C': 3})
        expect = np.array([expect])
        actual = ext(seqs)
        npt.assert_array_equal(actual, expect)

        seqs = self._translate_seqs(['AGTGGGTTCCC',
                                     'GGGGGGGGGGG'])
        expect = []
        expect.append(self._freq({'A': 1, 'G': 4, 'T': 3, 'C': 3}))
        expect.append(self._freq({'G': 11}))
        expect = np.array(expect)
        actual = ext(seqs)
        npt.assert_array_equal(actual, expect)

    def test_k4(self):
        ext = fe.KmersFeatureExtractor(4)

        seqs = self._translate_seqs('AAAA')
        expect = self._freq({'AAAA': 1})
        expect = np.array([expect])
        actual = ext(seqs)
        npt.assert_array_equal(actual, expect)

        seqs = self._translate_seqs('AAAAAAAA')
        expect = self._freq({'AAAA': 5})
        expect = np.array([expect])
        actual = ext(seqs)
        npt.assert_array_equal(actual, expect)

        seqs = self._translate_seqs(['AAAAAA',
                                     'CGCGCG'])
        expect = []
        expect.append(self._freq({'AAAA': 3}))
        expect.append(self._freq({'CGCG': 2,
                                  'GCGC': 1}))
        expect = np.array(expect)
        actual = ext(seqs)
        assert actual.shape == (2, 4**4)
        npt.assert_array_equal(actual, expect)
