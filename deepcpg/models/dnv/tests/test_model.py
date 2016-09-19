import keras.activations as kact

import predict.models.dnv.model as mod
import predict.models.dnv.params as pa


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
    pc.nb_filter = [1, 2, 3]
    pc.filter_len = [10, 11, 12]
    pc.pool_len = [2, 3, 4]
    pc.l1 = 0.1
    pc.l2 = 0.2

    p.target.nb_hidden = 0
    p.target.batch_norm = True

    m = mod.build(p, ['u0', 'u1'], seq_len=10, compile=False)

    assert 's_x' in m.input_order
    assert 'u0_y' in m.output_order
    assert 'u1_y' in m.output_order
    assert m.nodes['s_xd'].p == pc.drop_in
    for i in range(len(pc.nb_filter)):
        t = m.nodes['s_c%d' % (i + 1)]
        assert t.activation is kact.sigmoid
        assert t.nb_filter == pc.nb_filter[i]
        assert t.filter_length == pc.filter_len[i]
        assert t.W_regularizer.l1 == pc.l1
        assert t.W_regularizer.l2 == pc.l2
        assert m.nodes['s_p%d' % (i + 1)].pool_length == pc.pool_len[i]
    assert m.nodes['s_f1d'].p == pc.drop_out
    assert m.nodes['s_h1'].activation is kact.linear
    assert m.nodes['s_h1'].output_dim == pc.nb_hidden
    assert 's_h1b' in m.nodes
    assert m.nodes['s_h1a'].activation is kact.sigmoid
    assert m.nodes['s_h1d'].p == pc.drop_out
    assert 't_h1b' not in m.nodes


def test_all():
    p = pa.Params()

    p.seq = pa.SeqParams()
    p.seq.nb_filter = [1, 2]
    p.seq.filter_len = [10, 12]
    p.seq.pool_len = [2, 3]
    p.seq.nb_hidden = 2
    p.seq.activation = 'tanh'
    p.seq.drop_in = 0.1
    p.seq.drop_out = 0.5
    p.seq.batch_norm = False

    p.target = pa.TargetParams()
    p.target.activation = 'tanh'
    p.target.nb_hidden = 5
    p.target.drop_out = 0.0
    p.target.batch_norm = True

    p.optimizer = 'sgd'
    p.optimizer_params = {'lr': 0.05}

    targets = ['u0', 'u1']
    m = mod.build(p, targets, seq_len=10, compile=False)

    n = m.nodes

    assert n['s_xd'].p == p.seq.drop_in
    for i in range(len(p.seq.nb_filter)):
        t = n['s_c%d' % (i + 1)]
        assert t.nb_filter == p.seq.nb_filter[i]
        assert t.filter_length == p.seq.filter_len[i]
        assert t.activation is kact.tanh
        assert m.nodes['s_p%d' % (i + 1)].pool_length == p.seq.pool_len[i]
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
