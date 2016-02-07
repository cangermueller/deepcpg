import h5py as h5
import numpy as np
import numpy.testing as npt

from predict.models.dnn.data import encode_seqs, MAX_DIST


def test_data_shuffled():
    chromo = '5'

    f = h5.File('./data/data_shuffled.h5', 'r')
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


    t = np.array([5698714, 5698877, 5698955, 5699077, 5699268])
    assert np.all(cpos[:len(t)] == t)

    pos = 5698955
    i = np.nonzero(cpos == pos)
    assert len(i) == 1

    assert d['t0_y'][i] == 0.17974281
    assert d['t1_y'][i] == 0.13715278

    for k, v in d.items():
        if k.startswith('t'):
            assert np.all(v >= 0) & np.all(v <= 1)

    # Sequence
    delta = 5
    s = d['s_x'][i][0]
    o = s[s.shape[0] // 2 - delta: s.shape[0] // 2 + delta + 1]
    # s/\(\d\) \(\d\)/\1, \2/g
    e = np.array([[0, 1, 0, 0],
                  [1, 0, 0, 0],
                  [0, 0, 1, 0],
                  [1, 0, 0, 0],
                  [1, 0, 0, 0],
                  [0, 0, 0, 1],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1],
                  [0, 0, 1, 0],
                  [0, 1, 0, 0]], dtype='int8')
    npt.assert_array_equal(o, e)
