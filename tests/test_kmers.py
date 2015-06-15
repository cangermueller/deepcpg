import sys
import numpy.testing as npt
import os
import os.path as pt
import pandas as pd
import numpy as np
import pytest

from predict import kmers


class TestKmersExtractor(object):

    def test_translate(self):
        s = 'AGGTTCCCT'
        expect = [0, 1, 1, 2, 2, 3, 3, 3, 2]
        result = kmers.KmersExtractor().translate(s)
        npt.assert_array_equal(result, expect)

    def test_kmers(self):
        s = 'CGCGAT'
        expect = [3, 1, 3, 1, 0, 2]
        result = kmers.KmersExtractor(1).kmers(s)
        npt.assert_array_equal(result, expect)

        expect = [7, 13, 7, 1, 8]
        result = kmers.KmersExtractor(2).kmers(s)
        npt.assert_array_equal(result, expect)

        expect = [55, 29, 7, 33]
        result = kmers.KmersExtractor(3).kmers(s)
        npt.assert_array_equal(result, expect)

    def test_freq(self):
        s = 'CGCGAT'
        result = kmers.KmersExtractor(1).freq(s)

        s = 'CGCGAT'
        result = kmers.KmersExtractor(2).freq(s)
        assert len(result) == 16
        assert result[7] == 2
        assert result[13] == 1
        assert result[1] == 1
        assert result[8] == 1
        assert result[15] == 0

        s = 'CGCGC'
        result = kmers.KmersExtractor(3).freq(s)
        assert len(result) == 64
        assert result[55] == 2
        assert result[29] == 1

    def test_label(self):
        chrs = ['A', 'C', 'G', 'T']
        km = kmers.KmersExtractor(1, chrs=chrs)
        assert km.label(0) == 'A'
        assert km.label(1) == 'C'
        assert km.label(2) == 'G'
        assert km.label(3) == 'T'
        npt.assert_array_equal(km.labels(), chrs)

        km = kmers.KmersExtractor(2, chrs=chrs)
        assert km.label(0) == 'AA'
        assert km.label(1) == 'CA'
        assert km.label(2) == 'GA'
        assert km.label(3) == 'TA'
        assert km.label(4) == 'AC'
        assert km.label(5) == 'CC'
        assert km.label(6) == 'GC'
        assert km.label(7) == 'TC'
        assert km.label(15) == 'TT'
        labels = km.labels()
        assert len(labels) == 16
        assert labels[0] == 'AA'
        assert labels[15] == 'TT'

        km = kmers.KmersExtractor(3, chrs=chrs)
        assert km.label(0) == 'AAA'
        assert km.label(1) == 'CAA'
        assert km.label(63) == 'TTT'
        labels = km.labels()
        assert len(labels) == 64
        assert labels[0] == 'AAA'
        assert labels[63] == 'TTT'

        s = 'CGTACTC'
        km = kmers.KmersExtractor(4)
        expect = ['CGTA', 'GTAC', 'TACT', 'ACTC']
        result = km.labels(km.kmers(s))
        npt.assert_array_equal(result, expect)


def test_adjust_pos():
    seq = 'TTTTCGTTTT'
    assert 4 == kmers.adjust_pos(4, seq)
    assert 4 == kmers.adjust_pos(5, seq)
    assert 4 == kmers.adjust_pos(3, seq)
    assert kmers.adjust_pos(2, seq) is None
    assert kmers.adjust_pos(6, seq) is None


class TestApp(object):

    def __hdf_equal(self, file1, path1, file2, path2):
        h1 = pd.read_hdf(file1, path1)
        h2 = pd.read_hdf(file2, path2)
        return np.all(h1.values == h2.values)

    @pytest.mark.skipif(True, reason='')
    def test_run(self):
        data_dir = 'data/test_kmers/kmers_extractor'
        pos_file = pt.join(data_dir, 'pos.txt')
        seq_file = pt.join(data_dir, 'seq_file.h5')
        out_file = pt.join(data_dir, 'out_file.h5')
        ref_file = pt.join(data_dir, 'out_file0.h5')
        cmd = '../predict/kmers.py %s %s --out_file %s --wlen 5 --k 2'
        cmd = cmd % (seq_file, pos_file, out_file)
        assert os.system(cmd) == 0
        assert self.__hdf_equal(out_file, '/', ref_file, '/kmers')
        os.remove(out_file)
