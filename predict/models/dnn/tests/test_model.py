from predict.models.dnn_mc_cpg_seq.model import Params, build

def test_params():
    p = Params()
    p.cpg.dropout_in = 0.0
    p.cpg.dropout_out = 0.5
    p.seq.update({'num_hidden': 0, 'batch_norm': True,
                  'optimizer_params': {'lr': 0.5}})
    print(p)

def test_seq():
    p = Params()
    p.seq.num_filter = 2
    p.seq.filter_len = 2
    p.seq.num_hidden = 0
    p.cpg = False
    p.target.num_hidden = 0

    m = build(p, ['u0'], compile=False)
