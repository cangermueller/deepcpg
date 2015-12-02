import h5py as h5
import numpy as np
import numpy.testing as npt
from predict.models.dnn.utils import DataReader, ArrayView

class TestDataReader(object):

    def make_file(self, path, chromos):
        f = h5.File(path, 'w')
        for chromo, l in chromos.items():
            f['/%s/pos' % (chromo)] = np.empty(l, dtype='bool')
        f.close()

    def test_reader(self):
        chromos = {'1': 10, '2': 5, '3': 3}
        path = 'data_reader.h5'
        self.make_file(path, chromos)

        print()
        r = DataReader(path, chunk_size=1, shuffle=False, loop=False)
        for chromo, i, j in r:
            print(chromo, i, j)


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









