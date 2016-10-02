import os
from glob import glob
import gzip as gz


class FastaSeq(object):

    def __init__(self, head, seq):
        self.head = head
        self.seq = seq


def parse_lines(lines):
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
    if gzip is None:
        gzip = filename.endswith('.gz')
    if gzip:
        lines = gz.open(filename, 'r').read().decode()
    else:
        lines = open(filename, 'r').read()
    lines = lines.splitlines()
    return parse_lines(lines)


def read_chromo(dna_db, chromo):
    filename = glob(os.path.join(dna_db, '*.chromosome.%s.*fa.gz' % chromo))
    if len(filename) != 1:
        raise 'File for chromosome "%s" not found in "%s"!' (chromo, dna_db)
    filename = filename[0]

    fasta_seqs = read_file(filename)
    if len(fasta_seqs) != 1:
        raise 'Single sequence expected in file "%s"!' % filename
    return fasta_seqs[0].seq
