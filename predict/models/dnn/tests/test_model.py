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
    pc.nb_filter = [1, 2, 3]
    pc.filter_len = [10, 11, 12]
    pc.pool_len = [2, 3, 4]
    pc.l1 = 0.1
    pc.l2 = 0.2

    p.target.nb_hidden = 2
    p.target.batch_norm = True

    m = mod.build(p, ['u0', 'u1'], cpg_len=10, compile=False)

    assert 'c_x' in m.input_order
    assert 'u0_y' in m.output_order
    assert 'u1_y' in m.output_order
    assert m.nodes['c_xd'].p == pc.drop_in
    for i in range(len(pc.nb_filter)):
        t = m.nodes['c_c%d' % (i + 1)]
        assert t.activation is kact.sigmoid
        assert t.nb_filter == pc.nb_filter[i]
        assert t.nb_row == 1
        assert t.nb_col == pc.filter_len[i]
        assert t.W_regularizer.l1 == pc.l1
        assert t.W_regularizer.l2 == pc.l2
        assert m.nodes['c_p%d' % (i + 1)].pool_size == (1, pc.pool_len[i])
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
    pc.nb_filter = [1, 2, 3]
    pc.filter_len = [10, 11, 12]
    pc.pool_len = [2, 3, 4]
    pc.l1 = 0.1
    pc.l2 = 0.2

    p.target.nb_hidden = 0
    p.target.batch_norm = True

    m = mod.build(p, ['u0', 'u1'], seq_len=10, compile=True)

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

    p.cpg = pa.CpgParams()
    p.cpg.filter_len = [1, 2]
    p.cpg.nb_filter = [3, 4]
    p.cpg.pool_len = [5, 6]
    p.cpg.activation = 'sigmoid'
    p.cpg.drop_in = 0.0
    p.cpg.drop_out = 0.0
    p.cpg.nb_hidden = 128
    p.cpg.batch_norm = True

    p.seq = pa.SeqParams()
    p.seq.filter_len = [1, 2]
    p.seq.nb_filter = [3, 4]
    p.seq.pool_len = [5, 6]
    p.seq.activation = 'tanh'
    p.seq.drop_in = 0.1
    p.seq.drop_out = 0.5
    p.seq.nb_hidden = 64
    p.seq.batch_norm = False

    p.joint = pa.JointParams()
    p.joint.activation = 'linear'
    p.joint.nb_hidden = 32
    p.joint.drop_out = 0.1
    p.joint.batch_norm = True

    p.target = pa.TargetParams()
    p.target.activation = 'tanh'
    p.target.nb_hidden = 5
    p.target.drop_out = 0.0
    p.target.batch_norm = True

    p.optimizer = 'sgd'
    p.optimizer_params = {'lr': 0.05}

    targets = ['u0', 'u1']
    m = mod.build(p, targets, seq_len=10, cpg_len=5, compile=True)

    n = m.nodes

    assert 'c_xd' not in n

    pc = p.cpg
    for i in range(len(pc.nb_filter)):
        t = m.nodes['c_c%d' % (i + 1)]
        assert t.activation is kact.sigmoid
        assert t.nb_filter == pc.nb_filter[i]
        assert t.nb_row == 1
        assert t.nb_col == pc.filter_len[i]
        assert t.W_regularizer.l1 == pc.l1
        assert t.W_regularizer.l2 == pc.l2
        assert m.nodes['c_p%d' % (i + 1)].pool_size == (1, pc.pool_len[i])
    assert n['c_h1'].output_dim == pc.nb_hidden
    assert n['c_h1'].activation is kact.linear
    assert 'c_h1b' in n
    assert n['c_h1a'].activation is kact.sigmoid
    assert 'c_h1d' not in n

    assert n['s_xd'].p == p.seq.drop_in
    ps = p.seq
    for i in range(len(ps.nb_filter)):
        t = n['s_c%d' % (i + 1)]
        assert t.nb_filter == ps.nb_filter[i]
        assert t.filter_length == ps.filter_len[i]
        assert t.activation is kact.tanh
        assert m.nodes['s_p%d' % (i + 1)].pool_length == ps.pool_len[i]
    assert n['s_f1d'].p == ps.drop_out
    assert n['s_h1'].activation is kact.linear
    assert n['s_h1'].output_dim == ps.nb_hidden
    assert 's_h1b' not in n
    assert n['s_h1a'].activation is kact.tanh
    assert n['s_h1d'].p == ps.drop_out

    assert n['j_h1'].output_dim == p.joint.nb_hidden
    assert n['j_h1'].activation is kact.linear
    assert 'j_h1b' in n
    assert n['j_h1a'].activation is kact.linear
    assert n['j_h1d'].p == p.joint.drop_out

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


def test_outputs():
    p = pa.Params()
    p.cpg = False
    p.seq = pa.CpgParams()
    ps = p.seq
    ps.nb_filter = [2]
    ps.filter_len = [4]
    ps.pool_len = [2]
    ps.activation = 'sigmoid'
    ps.nb_hidden = 0
    ps.drop_in = 0.5
    ps.drop_out = 0.1

    targets = ['c0', 'c1', 'u0', 'u1', 's0', 's1']
    model = mod.build(p, targets, 10, compile=True)
    for target in targets:
        no = '%s_o' % (target)
        ny = '%s_y' % (target)
        assert no in model.nodes
        assert ny in model.outputs
        assert model.nodes[no] is model.outputs[ny]
        assert model.nodes[no].activation is kact.sigmoid
    assert len(model.output_order) == len(targets)
    if hasattr(model, 'loss'):
        assert len(model.loss) == len(targets)
        for target in targets:
            ny = '%s_y' % (target)
            if target.startswith('s'):
                assert model.loss[ny] == 'rmse'
            else:
                assert model.loss[ny] == 'binary_crossentropy'
