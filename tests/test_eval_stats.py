import os.path as pt
import sys
import numpy as np
from numpy import nan
import numpy.testing as npt
import pandas as pd

__dir = pt.dirname(pt.realpath(__file__))
sys.path.insert(0, pt.join(__dir, '../predict'))
import eval_stats as es


def test_stats():
    d = np.array([[1, 0, nan],      # 2 ||
                  [nan, nan, 1],    # 4 |||
                  [nan, 1, 1],      # 5  ||
                  [0, 1, 0]])       # 8    |
    d = pd.DataFrame(d, index=[2, 4, 5, 8])

    delta = 2
    cpg_cov = [2/3, 1/3, 2/3, 3/3]
    win_cpg_cov = [np.mean(cpg_cov[0:2]), np.mean(cpg_cov[0:3]), np.mean(cpg_cov[1:3]), cpg_cov[3]]
    win_var = [np.var([1, 0, 1]), np.var([1, 0.5, 1]), np.var([1, 1]), np.var([0, 1, 0])]
    min_dist = [np.mean([6, 3]), 1, np.mean([3, 1]), np.mean([6, 3, 3])]
    wlen = 2 * delta + 1
    dens = [2/wlen, 3/wlen, 2/wlen, 1/wlen]

    s = es.stats(d, delta)
    npt.assert_array_almost_equal(s.cpg_cov, cpg_cov)
    npt.assert_array_almost_equal(s.win_cpg_cov, win_cpg_cov)
    npt.assert_array_almost_equal(s.win_var, win_var)
    npt.assert_array_almost_equal(s.min_dist, min_dist)
    npt.assert_array_almost_equal(s.win_cpg_density, dens)

    d['pos'] = d.index
    d['chromo'] = '1'
    d = d.set_index(['chromo', 'pos'])
    s = es.stats_all(d, delta)
    npt.assert_array_almost_equal(s.cpg_cov, cpg_cov)
    npt.assert_array_almost_equal(s.win_cpg_cov, win_cpg_cov)
    npt.assert_array_almost_equal(s.win_var, win_var)
    npt.assert_array_almost_equal(s.min_dist, min_dist)
    npt.assert_array_almost_equal(s.win_cpg_density, dens)
