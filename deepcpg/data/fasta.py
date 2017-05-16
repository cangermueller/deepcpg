"""Functions reading FASTA files."""

from __future__ import division
from __future__ import print_function

import os
from glob import glob
import gzip as gz

from six.moves import range

from ..utils import to_list


class FastaSeq(object):
    """FASTA sequence."""

    def __init__(self, head, seq):
        self.head = head
        self.seq = seq


def parse_lines(lines):
    """Parse FASTA sequences from list of strings.

    Parameters
    ----------
    lines: list
        List of lines from FASTA file.

    Returns
    -------
    list
        List of :class:`FastaSeq` objects.
    """
    seqs = []
    seq = None
    start = None
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if len(line) > 0]
    for i in range(len(lines)):
        if lines[i][0] == '>':
            if start is not None:
                head = lines[start]
                seq = ''.join(lines[start + 1: i])
                seqs.append(FastaSeq(head, seq))
            start = i
    if start is not None:
        head = lines[start]
        seq = ''.join(lines[start + 1:])
        seqs.append(FastaSeq(head, seq))
    return seqs


def read_file(filename, gzip=None):
    """Read FASTA file and return sequences.

    Parameters
    ----------
    filename: str
        File name.
    gzip: bool
        If `True`, file is gzip compressed. If `None`, suffix is used to
        determine if file is compressed.

    Returns
    -------
        List of :class:`FastaSeq` objects.
    """
    list
    if gzip is None:
        gzip = filename.endswith('.gz')
    if gzip:
        lines = gz.open(filename, 'r').read().decode()
    else:
        lines = open(filename, 'r').read()
    lines = lines.splitlines()
    return parse_lines(lines)


def select_file_by_chromo(filenames, chromo):
    """Select file of chromosome `chromo`.

    Parameters
    ----------
    filenames: list
        List of file names or directory with FASTA files.
    chromo: str
        Chromosome that is selected.

    Returns
    -------
    str
        Filename in `filenames` that contains chromosome `chromo`.
    """
    filenames = to_list(filenames)
    if len(filenames) == 1 and os.path.isdir(filenames[0]):
        filenames = glob(os.path.join(filenames[0],
                                      '*.dna.chromosome.%s.fa*' % chromo))

    for filename in filenames:
        if filename.find('chromosome.%s.fa' % chromo) >= 0:
            return filename


def read_chromo(filenames, chromo):
    """Read DNA sequence of chromosome `chromo`.

    Parameters
    ----------
    filenames: list
        List of FASTA files.
    chromo: str
        Chromosome that is read.

    Returns
    -------
    str
        DNA sequence of chromosome `chromo`.
    """
    filename = select_file_by_chromo(filenames, chromo)
    if not filename:
        raise ValueError('DNA file for chromosome "%s" not found!' % chromo)

    fasta_seqs = read_file(filename)
    if len(fasta_seqs) != 1:
        raise ValueError('Single sequence expected in file "%s"!' % filename)
    return fasta_seqs[0].seq
