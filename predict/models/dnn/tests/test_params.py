import numpy as np
import scipy.stats as sps

import predict.models.dnn.params as pa


def test_params():
    seq = {
        'drop_in': 0.1,
        'drop_out': 0.0,
        'nb_hidden': 0,
        'batch_norm': True,
    }
    cpg = {
        'drop_in': 0.0,
        'drop_out': 0.5,
        'nb_hidden': 10,
        'batch_norm': False
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
        'target': target,
        'optimizer': 'RMSprop',
        'optimizer_params': {'lr': 0.5},
        'foo': 'bar'
    })
    for k, v in seq.items():
        assert vars(p.seq)[k] == v
    for k, v in cpg.items():
        assert vars(p.cpg)[k] == v
    for k, v in target.items():
        assert vars(p.target)[k] == v
    assert p.optimizer == 'RMSprop'
    assert p.optimizer_params['lr'] == 0.5
    assert 'foo' not in vars(p)

    p.update({'cpg': False, 'seq': False})
    assert p.cpg is False
    assert p.seq is False

    p.update({'cpg': cpg, 'seq': False})
    for k, v in cpg.items():
        assert vars(p.cpg)[k] == v
    assert p.seq is False

    p = pa.Params()
    p.update({'cpg': cpg, 'seq': False})
    for k, v in cpg.items():
        assert vars(p.cpg)[k] == v
    assert p.seq is False


def test_params_validate():
    p = pa.Params()
    p.cpg.nb_filter = 3
    p.cpg.pool_len = 5
    p.seq.nb_filter = 1
    p.seq.pool_len = 10
    p.seq.nb_hidden = 10
    p.cpg.nb_hidden = 3
    p.target.nb_hidden = 5
    p.validate()
    assert p.target.nb_hidden == 3
    assert p.cpg.pool_len == 3
    assert p.seq.pool_len == 1


def test_param_sampler():
    np.random.seed(0)
    p = {'cpg': False,
         'seq': {
             'nb_filter': sps.randint(4, 16),
             'filter_len': sps.randint(2, 10),
             'drop_out': sps.uniform(0, 0.5)
         },
         'target': {
             'nb_hidden': [10, 20, 30],
             'drop_out': sps.uniform(0, 0.1)
         },
         'optimizer': ['adam', 'rmsprop', 'sgd'],
         'optimizer_params': {'lr': [0.01, 0.001, 0.0001]},
         'batch_norm': [True, False],
         'activation': ['relu', 'tanh']
         }
    for sample in pa.ParamSampler(p, 20):
        assert sample.cpg is False
        assert sample.seq.nb_filter >= 4
        assert sample.seq.nb_filter < 16
        assert sample.seq.filter_len >= 2
        assert sample.seq.filter_len < 10
        assert sample.target.nb_hidden <= sample.seq.nb_hidden
        if sample.target.nb_hidden < sample.seq.nb_hidden:
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
