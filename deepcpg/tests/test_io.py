import numpy as np
import numpy.testing as npt

from predict.io import sort_cpos


def test_sort_cpos():
    d = dict(
        chromo=np.arange(10)[::-1],
        pos=np.arange(100, 110)[::-1],
        x=np.arange(200, 210)[::-1]
        )
    ds = sort_cpos(d)
    for k, v in d.items():
        npt.assert_array_equal(ds[k], np.array(v[::-1]))

    d = dict(
        chromo=[5, 5, 2, 1, 2, 5, 1],
        pos=   [9, 2, 7, 5, 4, 1, 0],
        x=     [1, 2, 3, 4, 5, 6, 7]
        )
    e = dict(
        chromo=[1, 1, 2, 2, 5, 5, 5],
        pos   =[0, 5, 4, 7, 1, 2, 9],
        x     =[7, 4, 5, 3, 6, 2, 1]
        )
    ds = sort_cpos(d)
    for k, v in e.items():
        npt.assert_array_equal(ds[k], np.array(v))
