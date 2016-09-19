import numpy as np
import scipy.stats as sps
import sys

import predict.models.dnn.params as pa


def test_params():
    seq = {
        'nb_filter': [1, 2, 3],
        'filter_len': [10, 12, 13],
        'pool_len': [3, 4, 5],
        'drop_in': 0.1,
        'drop_out': 0.0,
        'nb_hidden': 0,
        'batch_norm': True,
        'l1': 0,
        'l2': 0.5
    }

    cpg = {
        'nb_filter': [3, 2, 1],
        'filter_len': [13, 12, 11],
        'pool_len': [5, 3, 3],
        'drop_in': 0.0,
        'drop_out': 0.4,
        'nb_hidden': 10,
        'batch_norm': True,
        'l1': 0.1,
        'l2': 0.2
    }

    joint = {
        'nb_hidden': 9,
        'drop_out': 0.4,
        'batch_norm': True
    }

    target = {
        'nb_hidden': 10,
        'drop_out': 0.5,
        'batch_norm': True
    }

    p = pa.Params()
    p.update({
        'cpg': cpg,
        'seq': seq,
        'joint': joint,
        'target': target,
        'optimizer': 'RMSprop',
        'optimizer_params': {'lr': 0.5},
        'foo': 'bar',
        'batch_size': 256
    })

    for k, v in seq.items():
        assert vars(p.seq)[k] == v
    for k, v in cpg.items():
        assert vars(p.cpg)[k] == v
    for k, v in joint.items():
        assert vars(p.joint)[k] == v
    for k, v in target.items():
        assert vars(p.target)[k] == v
    assert p.optimizer == 'RMSprop'
    assert p.optimizer_params['lr'] == 0.5
    assert p.batch_size == 256
    assert 'foo' not in vars(p)

    p.update({'cpg': False, 'seq': False})
    assert p.cpg is False
    assert p.seq is False

    p.update({'cpg': cpg, 'seq': False})
    for k, v in cpg.items():
        assert vars(p.cpg)[k] == v
    assert p.seq is False

    p.update({'cpg': cpg, 'seq': seq, 'joint': False})
    assert p.cpg is not False
    assert p.seq is not False
    assert p.joint is False


def test_params_validate():
    p = pa.Params()
    p.cpg.nb_filter = 3
    p.cpg.pool_len = 5
    p.cpg.nb_hidden = 3

    p.seq.nb_filter = 1
    p.seq.pool_len = 10
    p.seq.nb_hidden = 10

    p.joint = pa.JointParams()
    p.joint.nb_hidden = 20

    p.target.nb_hidden = 5
    p.validate(nb_hidden=True)
    assert p.joint.nb_hidden == 3
    assert p.target.nb_hidden == 3
    assert p.cpg.pool_len[0] == 3
    assert p.seq.pool_len[0] == 1


def test_param_sampler():
    np.random.seed(0)
    p = {'cpg': False,
         'seq': {
             'nb_filter': [sps.randint(4, 16), sps.randint(20, 30)],
             'filter_len': [sps.randint(1, 4), sps.randint(5, 10)],
             'pool_len': [[2, 2]],
             'drop_out': sps.uniform(0, 0.5),
             'nb_hidden': [0, 10, 20]
         },
         'joint': {
             'nb_hidden': [10, 20, 40]
         },
         'target': {
             'nb_hidden': [20, 40, 50],
             'drop_out': sps.uniform(0, 0.1)
         },
         'optimizer': ['adam', 'rmsprop', 'sgd'],
         'optimizer_params': {'lr': [0.01, 0.001, 0.0001]},
         'batch_norm': [True, False],
         'activation': ['relu', 'tanh']
         }
    for sample in pa.ParamSampler(p, 20):
        assert sample.cpg is False
        assert sample.seq.nb_filter[0] >= 4
        assert sample.seq.nb_filter[0] < 16
        assert sample.seq.filter_len[0] >= 1
        assert sample.seq.filter_len[0] < 4
        assert sample.seq.nb_filter[1] >= 20
        assert sample.seq.nb_filter[1] < 30
        assert sample.seq.filter_len[1] >= 5
        assert sample.seq.filter_len[1] < 10
        t = sys.maxsize
        if sample.seq.nb_hidden > 0:
            t = min(t, sample.seq.nb_hidden)
        if sample.joint.nb_hidden > 0:
            assert sample.joint.nb_hidden <= t
            if sample.joint.nb_hidden < t:
                assert sample.joint.nb_hidden in p['joint']['nb_hidden']
            t = min(t, sample.joint.nb_hidden)
        assert sample.target.nb_hidden <= t
        if sample.target.nb_hidden < t:
            assert sample.target.nb_hidden in p['target']['nb_hidden']
        assert sample.optimizer in p['optimizer']
        assert sample.optimizer_params['lr'] in p['optimizer_params']['lr']
        t = sample.seq.batch_norm
        assert t in p['batch_norm']
        assert sample.target.batch_norm == t
        if t:
            assert sample.optimizer.lower() == 'sgd'
        t = sample.seq.activation
        assert t in p['activation']
        assert sample.target.activation == t
        t = sample.seq.drop_out
        assert t >= 0.0
        assert t < 0.5
        t = sample.target.drop_out
        assert t >= 0.0
        assert t < 0.1
