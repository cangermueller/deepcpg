import os
import os

import numpy as np

from deepcpg.data import CPG_NAN
from deepcpg import models as mod


class TestModel(object):

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

    def test_reader(self):
        model = mod.Model()
        dna_wlen = 101
        cpg_wlen = 10
        output_names = ['cpg_BS27_4_SER', 'cpg_BS28_2_SER']
        class_weights = {'cpg_BS27_4_SER': {0: 0.3, 1: 0.7},
                         'cpg_BS28_2_SER': {0: 0.2, 1: 0.8}}
        replicate_names = ['BS27_4_SER', 'BS28_2_SER']
        reader = model.reader(self.data_files[0],
                              output_names=output_names,
                              dna_wlen=dna_wlen,
                              replicate_names=replicate_names,
                              cpg_wlen=cpg_wlen,
                              class_weights=class_weights,
                              loop=False)

        for inputs, outputs, weights in reader:
            assert len(inputs) == 3
            dna = inputs['dna']
            assert dna.shape[1] == dna_wlen
            assert dna.shape[2] == 4
            tmp = dna.sum(axis=2)
            assert np.all((tmp == 0) | (tmp == 1))
            assert np.all(dna[:, dna_wlen // 2, 3] == 1)

            cpg_state = inputs['cpg/state']
            cpg_dist = inputs['cpg/state']
            assert np.all(cpg_state.shape == cpg_dist.shape)
            assert cpg_state.shape[1] == len(replicate_names)
            assert cpg_state.shape[2] == cpg_wlen
            np.all((cpg_state == CPG_NAN) | (cpg_state == 1) | (cpg_state == 1))
            idx = cpg_state == CPG_NAN
            assert np.all(cpg_dist[idx] == CPG_NAN)
            assert np.all((cpg_dist[~idx] >= 0) & (cpg_dist[~idx] <= 1))

            assert len(outputs) == len(weights)
            assert len(outputs) == len(output_names)
            for output_name in output_names:
                output = outputs[output_name]
                weight = weights[output_name]
                assert np.all((output == CPG_NAN) | (output == 0) |
                              (output == 1))
                assert np.all(weight[output == CPG_NAN] == 0)
                for cla in [0, 1]:
                    cw = class_weights[output_name][cla]
                    assert np.all(weight[output == cla] == cw)
