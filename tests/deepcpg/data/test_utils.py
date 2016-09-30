from collections import OrderedDict

from deepcpg import data


def test_h5_hnames_to_names():
    hnames = OrderedDict.fromkeys(['a', 'b'])
    names = data.h5_hnames_to_names(hnames)
    assert names == ['a', 'b']

    hnames = OrderedDict()
    hnames['a'] = ['a1', 'a2']
    hnames['b'] = ['b1', 'b2']
    names = data.h5_hnames_to_names(hnames)
    assert names == ['a/a1', 'a/a2', 'b/b1', 'b/b2']

    hnames = OrderedDict()
    hnames['a'] = 'a1'
    hnames['b'] = ['b1', 'b2']
    hnames['c'] = None
    names = data.h5_hnames_to_names(hnames)
    assert names == ['a/a1', 'b/b1', 'b/b2', 'c']
