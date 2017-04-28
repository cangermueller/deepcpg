from glob import glob
import os
from os import path as pt
from shutil import rmtree
import sys
from tempfile import mkdtemp

PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(PATH, '../../scripts'))

import dcpg_data
import dcpg_train


class TestTrain(object):

    def setup_class(self):
        self.in_dir = pt.join(PATH, 'data')
        self.cpg_dir = pt.join(self.in_dir, 'cpg_profiles')
        self.cpg_profiles = [pt.join(self.cpg_dir, 'BS27_4_SER.bed.gz'),
                             pt.join(self.cpg_dir, 'BS28_2_SER.bed.gz')]
        self.dna_dir = pt.join(self.in_dir, 'dna_db')

        self.data_dir = self.get_tmp_dir(self)
        self.make_data(self, self.data_dir, cpg_wlen=10, dna_wlen=101)

    def teardown_class(self):
        rmtree(self.data_dir)

    def get_tmp_dir(self):
        return mkdtemp(dir='/tmp', prefix='test_train_')

    def make_data(self, out_dir, cpg_wlen=None, dna_wlen=None):
        cmd = ['dcpg_data',
               '--out_dir', out_dir,
               '--chromos', 18, 19,
               '--nb_sample_chromo', 500,
               '--cpg_profiles'] + self.cpg_profiles
        if cpg_wlen:
            cmd += ['--cpg_wlen', cpg_wlen]
        if dna_wlen:
            cmd += ['--dna_files', self.dna_dir,
                    '--dna_wlen', dna_wlen]
        cmd = [str(arg) for arg in cmd]
        app = dcpg_data.App()
        assert app.run(cmd) == 0
        self.train_files = glob(pt.join(out_dir, 'c18_*.h5'))
        self.val_files = glob(pt.join(out_dir, 'c19_*.h5'))
        assert len(self.train_files)
        assert len(self.val_files)

    def _test_train(self, out_dir, args=None, nb_epoch=2):
        cmd = ['dcpg_train'] + self.train_files + \
            ['--val_files'] + self.val_files + \
            ['--out_dir', out_dir,
             '--nb_epoch', nb_epoch]
        if args:
            cmd += args
        cmd = [str(arg) for arg in cmd]
        app = dcpg_train.App()
        assert app.run(cmd) == 0
        must_exist = ['model.h5', 'model_weights_train.h5',
                      'model_weights_val.h5']
        for name in must_exist:
            assert os.path.isfile(pt.join(out_dir, name))

    def test_separately(self):
        # Test training models separately.
        model_dir = self.get_tmp_dir()
        cpg_model_dir = pt.join(model_dir, 'cpg')
        dna_model_dir = pt.join(model_dir, 'dna')
        joint_model_dir = pt.join(model_dir, 'joint')

        args = ['--cpg_model', 'RnnL1']
        self._test_train(cpg_model_dir, args=args)

        args = ['--dna_model', 'CnnL2h128']
        self._test_train(dna_model_dir, args=args)

        args = ['--cpg_model', cpg_model_dir,
                '--dna_model', dna_model_dir,
                '--joint_model', 'JointL2h512',
                '--fine_tune']
        self._test_train(joint_model_dir, args=args)

        rmtree(model_dir)

    def test_jointly(self):
        # Test training models jointly.
        model_dir = self.get_tmp_dir()

        args = ['--cpg_model', 'RnnL1',
                '--dna_model', 'CnnL2h128',
                '--joint_model', 'JointL2h512']
        self._test_train(model_dir, args=args)
