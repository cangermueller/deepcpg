import numpy as np


char2int_ = {'A': 0, 'G': 1, 'T': 2, 'C': 3, 'N': 4}
int2char_ = {v: k for k, v in char2int_.items()}

def char2int(seq):
    return [char2int_[x] for x in seq.upper()]

def int2char(seq, join=True):
    t = [int2char_[x] for x in seq]
    if join:
        t = ''.join(t)
    return t

def int2onehot(seqs, dim=4):
    """Special nucleotides will be encoded as [0, 0, 0, 0]."""
    seqs = np.atleast_2d(np.asarray(seqs))
    n = seqs.shape[0]
    l = seqs.shape[1]
    enc_seqs = np.zeros((n, l, dim), dtype='int8')
    for i in range(dim):
        t = seqs == i
        enc_seqs[t, i] = 1
    return enc_seqs

def onehot2int(seqs):
    return seqs.argmax(axis=2)
