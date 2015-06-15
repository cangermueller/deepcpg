import numpy as np
import numpy.testing as npt
import sys
import pytest
import os.path as pt

from predict import feature_extractor as fe


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


    def test_extract_k1(self):
        y = np.array([1, 3, 5, 8, 15])
        ys = np.array([0, 0, 1, 1, 0])

        x = np.array([2, 6, 10])
        expect = np.array([[0, 0, 1, 1],
                           [1, 1, 1, 2],
                           [1, 0, 2, 5]])
        result = fe.KnnCpgFeatureExtractor(1).extract(x, y, ys)
        npt.assert_array_equal(result, expect)

        x = np.array([0, 1, 3, 11, 15, 20])
        expect = np.array([[np.nan, 0, np.nan, 1],
                           [np.nan, 0, np.nan, 2],
                           [0, 1, 2, 2],
                           [1, 0, 3, 4],
                           [1, np.nan, 7, np.nan],
                           [0, np.nan, 5, np.nan]])
        result = fe.KnnCpgFeatureExtractor(1).extract(x, y, ys)
        npt.assert_array_equal(result, expect)

    def test_extract_k2(self):
        y = np.array([1, 3, 5, 8, 15])
        ys = np.array([0, 0, 1, 1, 0])
        x = np.array([2, 6, 10])
        expect = np.array([[np.nan, 0, 0, 1, np.nan, 1, 1, 3],
                           [0, 1, 1, 0, 3, 1, 2, 9],
                           [1, 1, 0, np.nan, 5, 2, 5, np.nan]])
        result = fe.KnnCpgFeatureExtractor(2).extract(x, y, ys)
        npt.assert_array_equal(result, expect)

        x = np.array([0, 1, 3, 8, 11, 15, 20])
        expect = np.array([[np.nan, np.nan, 0, 0, np.nan, np.nan, 1, 3],
                           [np.nan, np.nan, 0, 1, np.nan, np.nan, 2, 4],
                           [np.nan, 0, 1, 1, np.nan, 2, 2, 5],
                           [0, 1, 0, np.nan, 5, 3, 7, np.nan],
                           [1, 1, 0, np.nan, 6, 3, 4, np.nan],
                           [1, 1, np.nan, np.nan, 10, 7, np.nan, np.nan],
                           [1, 0, np.nan, np.nan, 12, 5, np.nan, np.nan]])
        result = fe.KnnCpgFeatureExtractor(2).extract(x, y, ys)
        npt.assert_array_equal(result, expect)

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
        npt.assert_array_equal(result, expect)


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
                      [10, 15], [10, 11], [11, 14], [14, 16]]
                      )
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
