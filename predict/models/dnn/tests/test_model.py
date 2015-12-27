import keras.activations as kact

import predict.models.dnn.model as mod
import predict.models.dnn.params as pa


def test_cpg():
    p = pa.Params()
    p.seq = False

    p.cpg = pa.CpgParams()
    pc = p.cpg
    pc.activation = 'sigmoid'
    pc.nb_hidden = 5
    pc.drop_in = 0.5
    pc.drop_out = 0.1
    pc.batch_norm = True
    pc.nb_filter = 1
    pc.filter_len = 2
    pc.pool_len = 4

    p.target.nb_hidden = 2
    p.target.batch_norm = True

    m = mod.build(p, ['u0', 'u1'], cpg_len=10, compile=False)

    assert 'c_x' in m.input_order
    assert 'u0_y' in m.output_order
    assert 'u1_y' in m.output_order
    assert m.nodes['c_xd'].p == pc.drop_in
    t = m.nodes['c_c1']
    assert t.activation is kact.sigmoid
    assert t.nb_filter == pc.nb_filter
    assert t.nb_row == 1
    assert t.nb_col == pc.filter_len
    assert m.nodes['c_p1'].pool_size == (1, pc.pool_len)
    assert m.nodes['c_f1d'].p == pc.drop_out
    assert m.nodes['c_h1'].activation is kact.linear
    assert m.nodes['c_h1'].output_dim == pc.nb_hidden
    assert m.nodes['c_h1a'].activation is kact.sigmoid
    assert m.nodes['c_h1d'].p == pc.drop_out
    assert 'c_h1b' in m.nodes

    assert 'u0_h1b' in m.nodes
    assert 'u1_h1b' in m.nodes


def test_seq():
    p = pa.Params()
    p.cpg = False

    p.seq = pa.SeqParams()
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
    p.target.batch_norm = True

    m = mod.build(p, ['u0', 'u1'], seq_len=10, compile=False)

    assert 's_x' in m.input_order
    assert 'u0_y' in m.output_order
    assert 'u1_y' in m.output_order
    assert m.nodes['s_xd'].p == pc.drop_in
    t = m.nodes['s_c1']
    assert t.activation is kact.sigmoid
    assert t.nb_filter == pc.nb_filter
    assert t.filter_length == pc.filter_len
    assert m.nodes['s_p1'].pool_length == pc.pool_len
    assert m.nodes['s_f1d'].p == pc.drop_out
    assert m.nodes['s_h1'].activation is kact.linear
    assert m.nodes['s_h1'].output_dim == pc.nb_hidden
    assert 's_h1b' in m.nodes
    assert m.nodes['s_h1a'].activation is kact.sigmoid
    assert m.nodes['s_h1d'].p == pc.drop_out
    assert 't_h1b' not in m.nodes


def test_joint():
    p = pa.Params()

    p.cpg = pa.CpgParams()
    p.cpg.nb_hidden = 3
    p.cpg.activation = 'sigmoid'
    p.cpg.drop_in = 0.0
    p.cpg.drop_out = 0.0
    p.cpg.batch_norm = True
    p.cpg.filter_len = 4
    p.cpg.nb_filter = 2

    p.seq = pa.SeqParams()
    p.seq.nb_hidden = 2
    p.seq.activation = 'tanh'
    p.seq.drop_in = 0.1
    p.seq.drop_out = 0.5
    p.seq.batch_norm = False
    p.seq.filter_len = 3
    p.seq.nb_filter = 1

    p.target = pa.TargetParams()
    p.target.activation = 'tanh'
    p.target.nb_hidden = 5
    p.target.drop_out = 0.0
    p.target.batch_norm = True

    p.optimizer = 'sgd'
    p.optimizer_params = {'lr': 0.05}

    targets = ['u0', 'u1']
    m = mod.build(p, targets, seq_len=10, cpg_len=5, compile=False)

    n = m.nodes

    assert 'c_xd' not in n
    t = n['c_c1']
    assert t.activation is kact.sigmoid
    assert t.nb_filter is p.cpg.nb_filter
    assert t.nb_row == 1
    assert t.nb_col is p.cpg.filter_len
    assert n['c_h1'].output_dim == p.cpg.nb_hidden
    assert n['c_h1'].activation is kact.linear
    assert 'c_h1b' in n
    assert n['c_h1a'].activation is kact.sigmoid
    assert 'c_h1d' not in n

    assert n['s_xd'].p == p.seq.drop_in
    t = n['s_c1']
    assert t.nb_filter is p.seq.nb_filter
    assert t.filter_length is p.seq.filter_len
    assert t.activation is kact.tanh
    assert n['s_f1d'].p == p.seq.drop_out
    assert n['s_h1'].activation is kact.linear
    assert n['s_h1'].output_dim == p.seq.nb_hidden
    assert 's_h1b' not in n
    assert n['s_h1a'].activation is kact.tanh
    assert n['s_h1d'].p == p.seq.drop_out

    for target in targets:
        def label(x):
            return '%s_%s' % (target, x)
        assert label('y') in m.output_order
        assert n[label('h1')].output_dim == p.target.nb_hidden
        assert n[label('h1')].activation is kact.linear
        assert label('h1b') in n
        assert n[label('h1a')].activation is kact.tanh
        assert label('h1d') not in n.keys()
        assert n[label('o')].activation is kact.sigmoid

    #  assert isinstance(m.optimizer, SGD)
    #  assert round(float(m.optimizer.lr.get_value()), 3) == p.optimizer_params['lr']
