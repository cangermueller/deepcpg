import numpy as np
import numpy.testing as npt

from predict.models.dnn.utils import ArrayView


class TestArrayView(object):

    def test_all(self):
        a = np.arange(40).reshape(10, -1)
        av = ArrayView(a)

        npt.assert_array_equal(av[0], a[0])
        npt.assert_array_equal(av[[1, 4, 5]], a[[1, 4, 5]])
        npt.assert_array_equal(av[3:9], a[3:9])
        npt.assert_array_equal(av[:], a[:])

    def test_range(self):
        a = np.arange(40).reshape(10, -1)

        av = ArrayView(a, start=5)
        assert len(av) == 5
        assert av.shape == (5, a.shape[1])
        npt.assert_array_equal(av[0], a[5])
        npt.assert_array_equal(av[[1, 3, 4]], a[[6, 8, 9]])
        npt.assert_array_equal(av[1:3], a[6:8])
        npt.assert_array_equal(av[:], a[5:])

        av = ArrayView(a, stop=5)
        assert len(av) == 5
        assert av.shape == (5, a.shape[1])
        npt.assert_array_equal(av[0], a[0])
        npt.assert_array_equal(av[[1, 2]], a[[1, 2]])
        npt.assert_array_equal(av[2:], a[2:5])
        npt.assert_array_equal(av[:], a[:5])

        av = ArrayView(a, start=3, stop=6)
        assert len(av) == 3
        assert av.shape == (3, a.shape[1])
        npt.assert_array_equal(av[0], a[3])
        npt.assert_array_equal(av[[0, 2]], a[[3, 5]])
        npt.assert_array_equal(av[2:], a[5:6])
        npt.assert_array_equal(av[:], a[3:6])

        av = ArrayView(a, start=8, stop=100)
        assert len(av) == 2
        assert av.shape == (2, 4)
        npt.assert_array_equal(av[:], a[8:])
        npt.assert_array_equal(av[0], a[8])
        npt.assert_array_equal(av[[0, 1]], a[[8, 9]])

        av = ArrayView(a, start=2, stop=5)
        assert len(av) == 3
        assert av.shape == (3, 4)
        npt.assert_array_equal(av[:, :], a[2:5, :])
        npt.assert_array_equal(av[:, 2], a[2:5, 2])
        npt.assert_array_equal(av[:, [1, -1]], a[2:5, [1, -1]])
        npt.assert_array_equal(av[:, 1:3], a[2:5, 1:3])
        npt.assert_array_equal(av[1:, 1:3], a[3:5, 1:3])
