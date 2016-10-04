import numpy as np


CHAR_TO_INT = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4}
INT_TO_CHAR = {v: k for k, v in CHAR_TO_INT.items()}

def char2int(seq):
    return [CHAR_TO_INT[x] for x in seq.upper()]

def int2char(seq, join=True):
    t = [INT_TO_CHAR[x] for x in seq]
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
