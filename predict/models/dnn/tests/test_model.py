from predict.models.dnn.model import *
from keras.activations import sigmoid, relu
import scipy.stats as sps


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

    p = Params()
    p.update({
        'cpg': cpg,
        'seq': seq,
        'target': target,
        'optimizer': 'RMSprop',
        'optimizer_params': {'lr': 0.5}
    })
    for k, v in seq.items():
        assert vars(p.seq)[k] == v
    for k, v in cpg.items():
        assert vars(p.cpg)[k] == v
    for k, v in target.items():
        assert vars(p.target)[k] == v
    assert p.optimizer == 'RMSprop'
    assert p.optimizer_params['lr'] == 0.5

    p.update({'cpg': False, 'seq': False})
    assert p.cpg == False
    assert p.seq == False

def test_params_validate():
    p = Params()
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
    for sample in ParamSampler(p, 20):
        assert sample.cpg == False
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


def test_seq():
    p = Params()
    p.cpg = False

    p.seq = SeqParams()
    pc = p.seq
    pc.activation = 'sigmoid'
    pc.nb_hidden = 5
    pc.drop_in = 0.5
    pc.drop_out = 0.1
    pc.batch_norm = True
    pc.nb_filter = 1
    pc.filter_len = 2
    pc.pool_len = 4

    p.target.nb_hidden = 0
    m = build(p, ['u0', 'u1'], seq_len=10, compile=False)
    assert 'u0_y' in m.output_order
    assert 'u1_y' in m.output_order
    assert 's_x' in m.input_order
    assert 's_h1b' in m.nodes.keys()
    assert m.nodes['s_h1d'].p == pc.drop_out
    assert m.nodes['s_xd'].p == pc.drop_in
    assert m.nodes['s_h1a'].activation is sigmoid
    t = m.nodes['s_c1']
    assert t.filter_length == pc.filter_len
    assert t.nb_filter == pc.nb_filter


def test_cpg():
    p = Params()
    p.seq = False

    p.cpg = CpgParams()
    pc = p.cpg
    pc.activation = 'sigmoid'
    pc.nb_hidden = 5
    pc.drop_in = 0.5
    pc.drop_out = 0.1
    pc.batch_norm = True
    pc.nb_filter = 1
    pc.filter_len = 2
    pc.pool_len = 4

    p.target.nb_hidden = 0

    m = build(p, ['u0', 'u1'], cpg_len=10, compile=False)
    assert 'u0_y' in m.output_order
    assert 'u1_y' in m.output_order
    assert 'c_x' in m.input_order
    assert 'c_h1b' in m.nodes.keys()
    assert m.nodes['c_h1d'].p == pc.drop_out
    assert m.nodes['c_xd'].p == pc.drop_in
    assert m.nodes['c_h1a'].activation is sigmoid
    t = m.nodes['c_c1']
    assert t.nb_filter == pc.nb_filter
    assert t.nb_row == 1
    assert t.nb_col == pc.filter_len


def test_joint():
    p = Params()
    p.seq = SeqParams()
    p.seq.nb_hidden = 2
    p.seq.activation = 'relu'
    p.seq.drop_in = 0.1
    p.seq.drop_out = 0.5
    p.seq.batch_norm = False
    p.seq.filter_len = 3
    p.seq.nb_filter = 1

    p.cpg = CpgParams()
    p.cpg.nb_hidden = 3
    p.cpg.activation = 'sigmoid'
    p.cpg.drop_in = 0.0
    p.cpg.drop_out = 0.0
    p.cpg.batch_norm = True
    p.cpg.filter_len = 4
    p.cpg.nb_filter = 2

    p.target = TargetParams()
    p.target.activation = 'sigmoid'
    p.target.nb_hidden = 2
    p.target.drop_out = 0.0
    p.target.batch_norm = True

    p.optimizer = 'sgd'
    p.optimizer_params = {'lr': 0.05}

    targets = ['u0', 'u1']
    m = build(p, targets, seq_len=10, cpg_len=5, compile=False)

    n = m.nodes

    for target in targets:
        def label(x):
            return '%s_%s' % (target, x)
        assert label('y') in m.output_order
        assert n[label('o')].activation is sigmoid
        assert label('h1d') not in n.keys()
        assert label('h1b') in n.keys()

    assert n['s_xd'].p == p.seq.drop_in
    assert n['s_f1d'].p == p.seq.drop_out
    assert n['s_h1d'].p == p.seq.drop_out
    assert n['s_c1'].activation is relu
    assert n['s_c1'].nb_filter is p.seq.nb_filter
    assert n['s_c1'].filter_length is p.seq.filter_len
    assert n['s_h1a'].activation is relu

    assert 'c_xd' not in n.keys()
    assert 'c_h1d' not in n.keys()
    assert n['c_c1'].activation is sigmoid
    assert n['c_c1'].nb_filter is p.cpg.nb_filter
    assert n['c_c1'].nb_row == 1
    assert n['c_c1'].nb_col is p.cpg.filter_len
    assert n['c_h1a'].activation is sigmoid

    #  assert isinstance(m.optimizer, SGD)
    #  assert round(float(m.optimizer.lr.get_value()), 3) == p.optimizer_params['lr']
