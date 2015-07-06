import argparse
import sys
import logging
import os.path as pt
import numpy as np
import warnings

from predict import hdf
from predict import data


class Processor(object):

    def __init__(self, out_path, out_group='/'):
        self.out_path = out_path
        self.out_group = out_group
        self.chromos = None
        self.pos_min = None
        self.pos_max = None
        self.nrows = None
        self.rng = np.random.RandomState(0)

    def write(self, d, sample):
        chromos = d.index.get_level_values('chromo').unique()
        for chromo in chromos:
            dc = d.loc[chromo]
            group = pt.join(self.out_group, 'cpg', str(chromo), sample)
            dc.to_hdf(self.out_path, group)

    def process(self, path):
        sample = pt.splitext(pt.basename(path))[0]
        d = data.read_cpg(path, self.chromos, self.nrows)
        if self.pos_min is not None:
            d = d.loc[d.pos >= self.pos_min]
        if self.pos_max is not None:
            d = d.loc[d.pos <= self.pos_max]
        d.set_index(['chromo', 'pos'], inplace=True)
        self.write(d, sample)
