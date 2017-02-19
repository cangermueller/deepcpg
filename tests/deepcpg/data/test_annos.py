from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.testing as npt
import pandas as pd

from deepcpg.data import annotations as annos


def test_join_overlapping():
    f = annos.join_overlapping

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


def test_in_which():
    f = annos.in_which
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


def test_is_in():
    ys = [2, 4, 12, 17]
    ye = [2, 8, 15, 18]

    x = [-1, 2, 2, 3, 4, 8, 15, 16]
    expect = [False, True, True, False, True, True, True, False]
    result = annos.is_in(x, ys, ye)
    npt.assert_array_equal(result, expect)


def test_distance():
    start = [3, 10, 17]
    end = [6, 15, 18]
    pos = [1, 2, 5, 8, 10, 15, 16, 19]
    expect = [2, 1, 0, 2, 0, 0, 1, 1]
    start = np.asarray(start)
    end = np.asarray(end)
    pos = np.asarray(pos)
    actual = annos.distance(pos, start, end)
    npt.assert_array_equal(actual, expect)

    pos = [1, 6, 7, 9]
    expect = [2, 0, 1, 1]
    start = np.asarray(start)
    end = np.asarray(end)
    pos = np.asarray(pos)
    actual = annos.distance(pos, start, end)
    npt.assert_array_equal(actual, expect)


def test_extend_frame():
    d = pd.DataFrame({
        'chromo': '1',
        'start': [2, 3, 3, 1, 1],
        'end':   [3, 3, 8, 2, 1]
    })
    d = d.loc[:, ['chromo', 'start', 'end']]
    expect = pd.DataFrame({
        'chromo': '1',
        'start':  [1, 2, 3, 1, 1],
        'end':    [4, 5, 8, 4, 4]
    })
    expect = expect.loc[:, ['chromo', 'start', 'end']]
    actual = annos.extend_len_frame(d, 4)
    npt.assert_array_equal(actual.values, expect.values)


def test_group_overlapping():
    npt.assert_array_equal(annos.group_overlapping([], []), [])
    npt.assert_array_equal(annos.group_overlapping([1], [2]), [0])
    s = [1, 5, 7, 11, 13, 16, 22]
    e = [3, 8, 9, 15, 17, 20, 24]
    g = [0, 1, 1,  2,  2,  2,  3]
    a = annos.group_overlapping(s, e)
    npt.assert_array_equal(a, g)
