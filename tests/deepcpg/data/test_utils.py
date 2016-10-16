from collections import OrderedDict
import os

import numpy as np
from numpy import testing as npt

from deepcpg import data as dat


def test_h5_hnames_to_names():
    hnames = OrderedDict.fromkeys(['a', 'b'])
    names = dat.h5_hnames_to_names(hnames)
    assert names == ['a', 'b']

    hnames = OrderedDict()
    hnames['a'] = ['a1', 'a2']
    hnames['b'] = ['b1', 'b2']
    names = dat.h5_hnames_to_names(hnames)
    assert names == ['a/a1', 'a/a2', 'b/b1', 'b/b2']

    hnames = OrderedDict()
    hnames['a'] = 'a1'
    hnames['b'] = ['b1', 'b2']
    hnames['c'] = None
    names = dat.h5_hnames_to_names(hnames)
    assert names == ['a/a1', 'b/b1', 'b/b2', 'c']


class TestH5reader(object):

    def setup(self):
        self.data_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            '../../integration_tests/data/data/')
        self.data_files = [
            os.path.join(self.data_path, 'c18_000000-005000.h5'),
            os.path.join(self.data_path, 'c18_005000-008712.h5'),
            os.path.join(self.data_path, 'c19_000000-005000.h5'),
            os.path.join(self.data_path, 'c19_005000-008311.h5')
        ]

    def test_h5_header(self):
        names = ['pos', 'chromo', '/outputs/cpg_BS27_4_SER']
        data_files = [self.data_files[0], self.data_files[-1]]
        batch_size = 15
        reader = dat.h5_reader(data_files,
                               names,
                               batch_size=batch_size,
                               loop=False,
                               shuffle=False)
        data = next(reader)
        assert np.all(data['chromo'][:5] == b'18')
        npt.assert_equal(data['pos'][:5],
                         [3000023, 3000086, 3000092, 3000103, 3000163])
        npt.assert_equal(data['/outputs/cpg_BS27_4_SER'][:5],
                         [1, 1, 1, -1, 0])

        nb_smaller = 0
        for data in reader:
            size = len(data['pos'])
            assert size <= batch_size
            if size < batch_size:
                assert nb_smaller == 0
                nb_smaller += 1
            else:
                nb_smaller = 0

        assert np.all(data['chromo'][-5:] == b'19')
        npt.assert_equal(data['pos'][-5:],
                         [4447803, 4447814, 4447818, 4447821, 4447847])
        npt.assert_equal(data['/outputs/cpg_BS27_4_SER'][-5:],
                         [1, 1, 1, 1, 1])

    def test_h5_read(self):
        data_files = [self.data_files[0], self.data_files[-1]]
        names = ['pos', 'chromo', '/outputs/cpg_BS27_4_SER']
        data = dat.h5_read(data_files, names, shuffle=False)

        assert np.all(data['chromo'][:5] == b'18')
        npt.assert_equal(data['pos'][:5],
                         [3000023, 3000086, 3000092, 3000103, 3000163])
        npt.assert_equal(data['/outputs/cpg_BS27_4_SER'][:5],
                         [1, 1, 1, -1, 0])

        assert np.all(data['chromo'][-5:] == b'19')
        npt.assert_equal(data['pos'][-5:],
                         [4447803, 4447814, 4447818, 4447821, 4447847])
        npt.assert_equal(data['/outputs/cpg_BS27_4_SER'][-5:],
                         [1, 1, 1, 1, 1])

    def test_nb_sample(self):
        """Test nb_sample together with shuffle. Should always return the same,
        if nb_sample < size of first data file."""

        names = ['pos', '/outputs/cpg_BS27_4_SER']
        data_ref = None
        nb_sample = 100  # Smaller than size of first file
        for i in range(10):
            data = dat.h5_read(self.data_files, names, shuffle=True,
                               nb_sample=nb_sample)
            if data_ref:
                for name in names:
                    assert len(data_ref[name]) == nb_sample
                    assert np.all(data_ref[name] == data[name])
            else:
                data_ref = data

    def test_read_reader(self):
        """Test if h5_read and h5_reader yield the same data."""
        nb_sample = 7777
        nb_loop = 10
        names = ['pos', 'chromo', '/outputs/cpg_BS27_4_SER']

        data = dat.h5_read(self.data_files, names, nb_sample=nb_sample)
        reader = dat.h5_reader(self.data_files, names, nb_sample=nb_sample,
                               loop=True)
        for loop in range(nb_loop):
            data_loop = dict()
            nb_sample_loop = 0
            while nb_sample_loop < nb_sample:
                data_batch = next(reader)
                for key, value in data_batch.items():
                    data_loop.setdefault(key, []).append(value)
                nb_sample_loop += len(value)
            assert nb_sample_loop == nb_sample
            for key, value in data_loop.items():
                fun = np.vstack if value[0].ndim > 1 else np.hstack
                data_loop[key] = fun(value)
                assert np.all(data[key] == data_loop[key])
