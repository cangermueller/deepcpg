import argparse
import sys
import logging
import pdb
import os.path as path
import pickle as pkl
import gzip as gz
import h5py

from predict import hdf


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
