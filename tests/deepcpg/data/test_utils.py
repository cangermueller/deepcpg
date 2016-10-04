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
