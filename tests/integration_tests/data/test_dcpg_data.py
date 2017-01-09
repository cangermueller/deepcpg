import os

import numpy as np
from numpy import testing as npt

from deepcpg.data import hdf, CPG_NAN
from deepcpg.data.fasta import read_chromo
from deepcpg.data.dna import CHAR_TO_INT


class TestMake(object):

    def setup_class(self):
        self.data_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'data')
        self.data_files = [
            os.path.join(self.data_path, 'c18_000000-005000.h5'),
            os.path.join(self.data_path, 'c18_005000-008712.h5'),
            os.path.join(self.data_path, 'c19_000000-005000.h5'),
            os.path.join(self.data_path, 'c19_005000-008311.h5')
        ]

        names = ['chromo', 'pos',
                 '/inputs/dna',
                 '/inputs/cpg/BS27_4_SER/dist',
                 '/inputs/cpg/BS27_4_SER/state',
                 '/inputs/cpg/BS28_2_SER/dist',
                 '/inputs/cpg/BS28_2_SER/state',
                 '/inputs/annos/exons',
                 '/inputs/annos/CGI',
                 '/outputs/cpg/BS27_4_SER',
                 '/outputs/cpg/BS28_2_SER',
                 '/outputs/stats/mean',
                 '/outputs/stats/var',
                 '/outputs/stats/cat_var',
                 '/outputs/stats/cat2_var',
                 '/outputs/stats/diff',
                 '/outputs/stats/mode',
                 '/outputs/bulk/BS9N_2I',
                 '/outputs/bulk/BS9N_SER'
                 ]
        self.data = hdf.read(self.data_files, names)
        self.chromo = self.data['chromo']
        self.pos = self.data['pos']

    def _test_outputs(self, name, expected):
        actual = self.data['/outputs/%s' % name]
        for e in expected:
            idx = (self.chromo == e[0].encode()) & (self.pos == e[1])
            assert idx.sum() == 1
            assert actual[idx] == e[2]

    def test_outputs(self):
        expected = [('18', 3000023, 1.0),
                    ('18', 3000086, 1.0),
                    ('18', 3012584, 0.0),
                    ('19', 4398070, 0.0),
                    ('19', 4428709, 1.0),
                    ('19', 4442494, 0.0),
                    ('19', 4447847, 1.0)
                    ]
        self._test_outputs('cpg/BS27_4_SER', expected)

        expected = [('18', 3000092, 1.0),
                    ('18', 3010064, 0.0),
                    ('18', 3140338, 1.0),
                    ('18', 3143169, 0.0),
                    ('19', 4187854, 1.0),
                    ('19', 4190571, 0.0),
                    ('19', 4192788, 0.0),
                    ('19', 4202077, 0.0)
                    ]
        self._test_outputs('cpg/BS28_2_SER', expected)

    def _test_dna(self, chromo):
        pos = self.pos[self.chromo == chromo.encode()]
        dna = self.data['/inputs/dna'][self.chromo == chromo.encode()]
        dna_wlen = dna.shape[1]
        center = dna_wlen // 2

        dna_seq = read_chromo(os.path.join(self.data_path, '../dna_db'),
                              chromo)

        idxs = np.linspace(0, len(pos) - 1, 100).astype(np.int32)
        for idx in idxs:
            p = pos[idx] - 1
            assert dna_seq[p:(p + 2)] == 'CG'
            assert dna[idx, center] == 3
            assert dna[idx, center + 1] == 2
            assert dna[idx, center + 10] == CHAR_TO_INT[dna_seq[p + 10]]
            assert dna[idx, center - 10] == CHAR_TO_INT[dna_seq[p - 10]]

    def test_dna(self):
        dna = self.data['/inputs/dna']
        dna_wlen = dna.shape[1]
        center = dna_wlen // 2
        assert np.all(dna[:, center] == 3)
        assert np.all(dna[:, (center + 1)] == 2)
        self._test_dna('18')
        self._test_dna('19')

    def _test_cpg_neighbors(self, name, expected):
        data_name = '/inputs/cpg/%s/' % name
        dist = self.data[data_name + 'dist'][100:-100]
        center = dist.shape[1] // 2
        assert np.all((dist[:, center - 2] > dist[:, center - 1]) |
                      (dist[:, center - 2] == CPG_NAN))
        assert np.all((dist[:, center] < dist[:, center + 1]) |
                      (dist[:, center + 1] == CPG_NAN))

        for exp in expected:
            pos = exp[1]
            idx = (self.chromo == exp[0].encode()) & (self.pos == pos)
            assert self.pos[idx] == pos
            assert np.sum(idx) == 1
            state = self.data[data_name + 'state'][idx].ravel()
            dist = self.data[data_name + 'dist'][idx].ravel()
            for i, left in enumerate(exp[2]):
                exp_dist = pos - left[0]
                exp_state = left[1]
                assert dist[center - i - 1] == exp_dist
                assert state[center - i - 1] == exp_state
            for i, right in enumerate(exp[3]):
                exp_dist = right[0] - pos
                exp_state = right[1]
                assert dist[center + i] == exp_dist
                assert state[center + i] == exp_state

    def test_cpg_neighbors(self):
        name = 'BS27_4_SER'
        expected = [
            ('18', 3000086,
             ((3000023, 1.0),),
             ((3000092, 1.0), (3000163, 0.0), (3000310, 1.0))
             ),
            ('18', 3000734,
             ((3000612, 1.0), (3000315, 1.0), (3000310, 1.0), (3000163, 0.0)),
             ((3000944, 0.0), (3001029, 0.0), (3001188, 0.0), (3004806, 1.0))
             ),
            ('18', 3047425,
             ((3047423, 1.0), (3046073, 0.0), (3046067, 1.0), (3046046, 0.0)),
             ((3047447, 0.0), (3047969, 0.0), (3047981, 0.0), (3047983, 1.0))
             ),
            ('19', 4364170,
             ((4363861, 1.0), (4362993, 1.0), (4359854, 1.0), (4359157, 1.0)),
             ((4372573, 1.0), (4376976, 1.0), (4377019, 1.0), (4378828, 1.0))
             ),
            ('19', 4428709,
             ((4410664, 1.0), (4407849, 0.0), (4406810, 1.0), (4406758, 1.0)),
             ((4429964, 1.0), (4429969, 1.0), (4430127, 1.0), (4430346, 0.0))
             ),
            ('19', 4447818,
             ((4447814, 1.0), (4447803, 1.0)),
             ((4447821, 1.0), (4447847, 1.0))
             )
        ]
        self._test_cpg_neighbors(name, expected)

        name = 'BS28_2_SER'
        expected = [
            ('18', 3010211,
             ((3010138, 1.0), (3010136, 0.0), (3010075, 1.0), (3010064, 0.0)),
             ((3010417, 1.0), (3010759, 1.0), (3012388, 1.0), (3012676, 1.0))
             ),
            ('18', 3039508,
             ((3038883, 1.0), (3038680, 0.0), (3038462, 1.0), (3031302, 0.0)),
             ((3039540, 1.0), (3039543, 1.0), (3039805, 1.0), (3039828, 1.0))
             ),
            ('19', 4201639,
             ((4201628, 0.0), (4201623, 0.0), (4201621, 1.0), (4201599, 0.0)),
             ((4201645, 0.0), (4201657, 0.0), (4201677, 0.0), (4201688, 0.0))
             ),
            ('19', 4185486,
             ((4185440, 1.0), (4184916, 1.0), (4184889, 0.0)),
             ((4185488, 0.0), (4186125, 0.0), (4187662, 1.0))
             ),
            ('19', 4201967,
             ((4201946, 0.0), (4201923, 0.0), (4201821, 0.0)),
             ((4201972, 0.0), (4202077, 0.0))
             )
        ]
        self._test_cpg_neighbors(name, expected)

    def _test_stats(self, chromo, pos, stat, value):
        idx = (self.chromo == chromo.encode()) & (self.pos == pos)
        stat = self.data['/outputs/stats/%s' % stat][idx]
        assert stat == value

    def test_stats(self):
        self._test_stats('18', 3010417, 'mean', 1.0)
        self._test_stats('18', 3010417, 'var', 0.0)
        self._test_stats('18', 3010417, 'diff', 0)
        self._test_stats('18', 3010417, 'mode', 1)
        self._test_stats('18', 3012173, 'mean', 0.0)
        self._test_stats('18', 3012173, 'var', 0.0)
        self._test_stats('18', 3012173, 'diff', 0)
        self._test_stats('18', 3012173, 'mode', 0)
        self._test_stats('18', 3052129, 'mean', 1.0)
        self._test_stats('18', 3052129, 'var', 0.0)
        self._test_stats('18', 3052129, 'diff', 0)
        self._test_stats('18', 3052129, 'mode', 1)
        self._test_stats('18', 3071630, 'mean', 0.5)
        self._test_stats('18', 3071630, 'var', 0.25)
        self._test_stats('18', 3071630, 'diff', 1)
        self._test_stats('18', 3071630, 'mode', 0)
        self._test_stats('19', 4201704, 'mean', 0.0)
        self._test_stats('19', 4201704, 'var', 0.0)
        self._test_stats('19', 4201704, 'diff', 0)
        self._test_stats('19', 4201704, 'mode', 0)
        self._test_stats('19', 4190571, 'mean', 0.5)
        self._test_stats('19', 4190571, 'var', 0.25)
        self._test_stats('19', 4190571, 'diff', 1)
        self._test_stats('19', 4190571, 'mode', 0)
        self._test_stats('19', 4190700, 'mean', 0.0)
        self._test_stats('19', 4190700, 'var', 0.0)
        self._test_stats('19', 4190700, 'diff', 0)
        self._test_stats('19', 4190700, 'mode', 0)

        v = self.data['/outputs/stats/var']
        assert np.all((v >= 0) & (v <= 0.25))

        cv = self.data['/outputs/stats/cat_var']
        assert np.all((cv == CPG_NAN) | (cv == 0) | (cv == 1) | (cv == 2))
        assert np.all(cv[v == CPG_NAN] == CPG_NAN)

        cv = self.data['/outputs/stats/cat2_var']
        assert np.all((cv == CPG_NAN) | (cv == 0) | (cv == 1))
        assert np.all(cv[v == CPG_NAN] == CPG_NAN)

    def _test_bulk(self, chromo, pos, name, expected):
        idx = (self.chromo == chromo.encode()) & (self.pos == pos)
        actual = float(self.data['/outputs/bulk/%s' % name][idx])
        npt.assert_almost_equal(actual, expected, 2)

    def test_bulk(self):
        self._test_bulk('18', 3000023, 'BS9N_2I', 0.0)
        self._test_bulk('18', 3000023, 'BS9N_SER', 0.75)
        self._test_bulk('18', 3000086, 'BS9N_2I', 0.0)
        self._test_bulk('18', 3000086, 'BS9N_SER', 0.666)
        self._test_bulk('18', 3004868, 'BS9N_2I', 0.042)
        self._test_bulk('18', 3004868, 'BS9N_SER', 0.1636)
        self._test_bulk('18', 3013979, 'BS9N_2I', -1)
        self._test_bulk('18', 3013979, 'BS9N_SER', 1.0)
        self._test_bulk('19', 4438754, 'BS9N_2I', -1)
        self._test_bulk('19', 4438754, 'BS9N_SER', 0.333)

    def _test_annos(self, chromo, pos, name, expected):
        idx = (self.chromo == chromo.encode()) & (self.pos == pos)
        actual = int(self.data['/inputs/annos/%s' % name][idx])
        assert actual == expected

    def test_annos(self):
        self._test_annos('18', 3000023, 'CGI', 0)
        self._test_annos('18', 3000023, 'exons', 0)
        self._test_annos('18', 3267095, 'exons', 1)
        self._test_annos('18', 4375924, 'exons', 1)
        self._test_annos('18', 5592169, 'exons', 1)
        self._test_annos('18', 5592176, 'exons', 1)
        self._test_annos('18', 5592182, 'exons', 1)
        self._test_annos('18', 5592199, 'exons', 1)
        self._test_annos('19', 4438754, 'exons', 0)
