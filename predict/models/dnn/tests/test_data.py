import h5py as h5
import numpy as np
import numpy.testing as npt
import os
import os.path as pt

from predict.models.dnn.data import encode_seqs, MAX_DIST


def test_encode_seqs():
    seqs = np.array([
        [0, 1, 4, 3],
        [1, 0, 3, 4]
    ])
    enc_seqs = np.array([
        [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        ], [
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
        ]
    ])
    got = encode_seqs(seqs)
    npt.assert_array_equal(enc_seqs, got)


def test_data():
    delta = 5
    pos = 3037343
    chromo = '5'

    f = h5.File('./data/data.h5', 'r')
    g = f['pos']
    dp = {k: g[k].value for k in g.keys()}
    dp['chromo'] = np.array([x.decode() for x in dp['chromo']])

    t = np.nonzero(dp['chromo'] == chromo)[0]
    s = t.min()
    e = t.max() + 1

    cpos = dp['pos'][s:e]

    k = np.argsort(cpos)
    cpos = cpos[k]

    assert np.all(cpos > 0)
    assert np.all(cpos[:-1] < cpos[1:])

    g = f['data']
    d = {x: g[x][s:e][k] for x in g.keys()}
    f.close()

    idx = np.nonzero(cpos == pos)
    assert len(idx) == 1
    idx = idx[0]

    # CpG targets
    assert d['c0_y'][idx] == 1

    # Stats targets
    stats_file = h5.File(pt.join(os.getenv('Cdata'), 'stats', '2i_ser.h5'), 'r')
    g = stats_file[chromo]
    s0 = g['ser_w3000_var'].value
    s1 = g['2i_w3000_var'].value
    sp = g['pos'].value
    stats_file.close()
    si = np.nonzero(sp == pos)[0]
    assert len(si) == 1
    sd = slice(idx - delta, idx + delta + 1)
    ss = slice(si - delta, si + delta + 1)
    npt.assert_array_almost_equal(d['s0_y'][sd].ravel(), s0[ss].ravel())
    npt.assert_array_almost_equal(d['s1_y'][sd].ravel(), s1[ss].ravel())

    # CpG neightbors sample 1
    c = d['c_x'][idx,:, 0, :][0]
    o = c[:, c.shape[1] // 2 - delta:c.shape[1] // 2 + delta + 4]
    e = np.array([[0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
               [ 1266, 1231, 42, 26, 7, 6127, 13551, 26418, 27492, 27495, 27497, 27503, 27519,
                27534]], dtype='float16')
    e[1] /= MAX_DIST
    npt.assert_array_almost_equal(o, e, decimal=3)

    # CpG neightbors sample 2
    c = d['c_x'][idx,:, 1, :][0]
    o = c[:, c.shape[1] // 2 - delta:c.shape[1] // 2 + delta + 4]
    e = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
               [ 8969, 8959, 8954, 7703, 5996, 9140, 10796, 14266, 26177, 26204, 26638, 26681, 26715,
                26767]], dtype='float16')
    e[1] /= MAX_DIST
    npt.assert_array_almost_equal(o, e, decimal=3)

    # Sequence
    s = d['s_x'][idx][0]
    o = s[s.shape[0] // 2 - delta: s.shape[0] // 2 + delta + 1]
    e = np.array([[ 0., 0., 0., 1.],
               [ 1., 0., 0., 0.],
               [ 0., 1., 0., 0.],
               [ 0., 1., 0., 0.],
               [ 0., 0., 0., 1.],
               [ 0., 0., 0., 1.],
               [ 0., 1., 0., 0.],
               [ 0., 1., 0., 0.],
               [ 0., 0., 0., 1.],
               [ 0., 0., 0., 1.],
               [ 0., 0., 0., 1.]], dtype='int8')
    npt.assert_array_equal(o, e)
