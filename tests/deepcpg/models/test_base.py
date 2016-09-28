from glob import glob
import os
import os.path as pt

import numpy as np
import h5py as h5

from deepcpg.data.preprocess import CPG_NAN
from deepcpg.models import base


def get_targets(data_file, target_filter=None):
    data_file = h5.File(data_file, 'r')
    targets = list(data_file['cpg'].keys())
    if target_filter is not None:
        if isinstance(target_filter, list):
            targets = [target for target in targets if target in target_filter]
        else:
            targets = targets[:target_filter]
    data_file.close()
    return targets


class TestDataGenerator(object):

    def setup(self):
        self.DATA_DIR = pt.join(os.getenv('Vcur'), 'data')

    def test_dna_cpg(self):
        test_files = glob(pt.join(self.DATA_DIR, '*.h5'))
        targets = get_targets(test_files[0])
        batch_size = 128
        data = base.data_generator(test_files, targets,
                                   dna_wlen=201, cpg_wlen=6,
                                   batch_size=batch_size)

        for i in range(10):
            x, y, w = next(data)
            assert len(x) == 1 + len(targets)
            dna = x[0]
            assert dna.shape[0] == batch_size
            assert dna.shape[1] == 201
            assert dna.shape[2] == 4
            assert np.all(dna.sum(axis=2) == 1)

            for j in range(len(targets)):
                cpg = x[j + 1]
                assert cpg.shape[0] == batch_size
                assert cpg.shape[1] == 6
                assert cpg.shape[2] == 2
                cpg_state = cpg[:, :, 0]
                assert np.all((cpg_state == 0) | (cpg_state == 1))
                cpg_dist = cpg[:, :, 1]
                assert np.all((cpg_dist > 0))
                assert np.all((cpg_dist <= 1))
                assert np.all(cpg_dist[:, 0] >= cpg_dist[:, 1])
                assert np.all(cpg_dist[:, -2] <= cpg_dist[:, -1])

            assert len(y) == len(targets)
            for j in range(len(y)):
                yj = y[j]
                wj = w[j]
                assert len(yj) == batch_size
                assert len(wj) == batch_size
                assert np.all(yj[wj == 0] == CPG_NAN)
                h = yj[wj != 0]
                assert np.all((h == 0) | (h == 1))
