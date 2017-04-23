from glob import glob
import os
from os import path as pt
from shutil import rmtree
import sys
from tempfile import mkdtemp

PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(PATH, '../../scripts'))

import dcpg_data


class TestData(object):

    def setup_class(self):
        self.in_dir = pt.join(PATH, 'data')
        self.cpg_dir = pt.join(self.in_dir, 'cpg_files')
        self.cpg_files = [pt.join(self.cpg_dir, 'BS27_4_SER.bed.gz'),
                          pt.join(self.cpg_dir, 'BS28_2_SER.bed.gz')]
        self.dna_dir = pt.join(self.in_dir, 'dna_db')

    def get_tmp_dir(self):
        return mkdtemp(dir='/tmp', prefix='test_data_')

    def _test_simple(self, args):
        out_dir = self.get_tmp_dir()
        cmd = ['dcpg_data',
               '--out_dir', out_dir,
               '--nb_sample', 1000,
               '--cpg_profiles'] + self.cpg_files + args
        cmd = [str(arg) for arg in cmd]
        app = dcpg_data.App()
        assert app.run(cmd) == 0
        assert len(glob(pt.join(out_dir, '*.h5')))
        rmtree(out_dir)

    def test_cpg_data(self):
        # Test extracting CpG neighbors.
        self._test_simple(['--cpg_wlen', 10])

    def test_dna_data(self):
        # Test extracting DNA windows.
        self._test_simple(['--dna_files', self.dna_dir,
                           '--dna_wlen', 101])

    def test_join_data(self):
        # Test extracting CpG neighbors and DNA windows.
        self._test_simple(['--cpg_wlen', 10,
                           '--dna_files', self.dna_dir,
                           '--dna_wlen', 101])
