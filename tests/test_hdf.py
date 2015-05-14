import sys
import os
import os.path as pt
import pandas as pd
import numpy as np
import tempfile as tmp
import pytest
import ipdb

__dir = pt.dirname(pt.realpath(__file__))
sys.path.insert(0, pt.join(__dir, '../biseq'))
import hdf

@pytest.fixture
def data_frame():
    return pd.DataFrame(np.random.rand(3, 2))

@pytest.fixture
def tmp_file(request, tmpdir):
    f = str(tmpdir.join('file.h5'))
    def tear_down():
        os.remove(f)
    request.addfinalizer(tear_down)
    return f

def test_ls(data_frame, tmp_file):
    data_frame.to_hdf(tmp_file, '/d')
    data_frame.to_hdf(tmp_file, '/g1/d')
    items = hdf.ls(tmp_file)
    assert items == ['d', 'g1']

def test_first_item(data_frame, tmp_file):
    data_frame.to_hdf(tmp_file, '/d')
    data_frame.to_hdf(tmp_file, '/g1/d')
    assert hdf.first_item(tmp_file) == 'd'

def test_split_path(data_frame, tmp_file):
    f, p = hdf.split_path(tmp_file)
    assert f == tmp_file
    assert p == '/'

    f, p = hdf.split_path(tmp_file + ':/g1/d')
    assert f == tmp_file
    assert p == '/g1/d'

    data_frame.to_hdf(tmp_file, '/d')
    f, p = hdf.split_path(tmp_file, first=True)
    assert f == tmp_file
    assert p == '/d'
