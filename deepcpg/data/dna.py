"""Functions for representing DNA sequences."""

from __future__ import division
from __future__ import print_function

from collections import OrderedDict

import numpy as np
from six.moves import range

# Mapping of nucleotides to integers
CHAR_TO_INT = OrderedDict([('A', 0), ('T', 1), ('G', 2), ('C', 3), ('N', 4)])
# Mapping of integers to nucleotides
INT_TO_CHAR = {v: k for k, v in CHAR_TO_INT.items()}


def get_alphabet(special=False, reverse=False):
    """Return char->int alphabet.

    Parameters
    ----------
    special: bool
        If `True`, remove special 'N' character.
    reverse: bool
        If `True`, return int->char instead of char->int alphabet.

    Returns
    -------
    OrderedDict
        DNA alphabet.
    """
    alpha = OrderedDict(CHAR_TO_INT)
    if not special:
        del alpha['N']
    if reverse:
        alpha = {v: k for k, v in alpha.items()}
    return alpha


def char_to_int(seq):
    """Translate chars of single sequence `seq` to ints.

    Parameters
    ----------
    seq: str
        DNA sequence.

    Returns
    -------
    list
        Integer-encoded `seq`.
    """
    return [CHAR_TO_INT[x] for x in seq.upper()]


def int_to_char(seq, join=True):
    """Translate ints of single sequence `seq` to chars.

    Parameters
    ----------
    seq: list
        Integers of sequences
    join: bool
        If `True` joint characters to `str`.

    Returns
    -------
    If `join=True`, `str`, otherwise list of chars.
    """
    t = [INT_TO_CHAR[x] for x in seq]
    if join:
        t = ''.join(t)
    return t


def int_to_onehot(seqs, dim=4):
    """One-hot encodes array of integer sequences.

    Takes array [nb_seq, seq_len] of integer sequence end encodes them one-hot.
    Special nucleotides (int > 4) will be encoded as [0, 0, 0, 0].

    Paramters
    ---------
    seqs: :class:`numpy.ndarray`
        [nb_seq, seq_len] :class:`numpy.ndarray` of integer sequences.
    dim: int
        Number of nucleotides

    Returns
    -------
    :class:`numpy.ndarray`
        [nb_seq, seq_len, dim] :class:`numpy.ndarray` of one-hot encoded
        sequences.
    """
    seqs = np.atleast_2d(np.asarray(seqs))
    n = seqs.shape[0]
    l = seqs.shape[1]
    enc_seqs = np.zeros((n, l, dim), dtype='int8')
    for i in range(dim):
        t = seqs == i
        enc_seqs[t, i] = 1
    return enc_seqs


def onehot_to_int(seqs, axis=-1):
    """Translates one-hot sequences to integer sequences."""
    return seqs.argmax(axis=axis)
