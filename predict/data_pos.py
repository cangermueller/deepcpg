import argparse
import sys
import logging
import os.path as pt
import pandas as pd
import warnings

from predict import hdf
from predict import data


def compute_pos(path, dataset, chromo):
    pos = set()
    group = pt.join(dataset, 'cpg', (chromo))
    samples = hdf.ls(path, group)
    for sample in samples:
        d = pd.read_hdf(path, pt.join(group, sample))
        pos.update(d.index.values)
    return sorted(list(pos))

def add_pos(path, dataset):
    chromos = hdf.ls(path, pt.join(dataset, 'cpg'))
    for chromo in chromos:
        p = compute_pos(path, dataset, chromo)
        p = pd.Series(p)
        group = pt.join(dataset, 'pos', chromo)
        p.to_hdf(path, group)
