import argparse
import sys
import logging
import os.path as pt
import numpy as np
import warnings

from predict import hdf
from predict import data


class Processor(object):

    def __init__(self, out_path, out_group='/', test_size=0.5, val_size=0.1):
        self.out_path = out_path
        self.out_group = out_group
        self.test_size = test_size
        self.val_size = val_size
        self.chromo = None
        self.pos_min = None
        self.pos_max = None
        self.nrows = None
        self.rng = np.random.RandomState(0)

    def write(self, d, name, sample):
        for chromo in d.chromo.unique():
            dc = d.loc[d.chromo == chromo]
            dc = dc.loc[:, ~dc.columns.isin(['chromo'])]
            group = pt.join(self.out_group, name, 'cpg', str(chromo), sample)
            dc.to_hdf(self.out_path, group, format='t', data_columns=True)

    def split(self, d, size_b=0.5):
        """Splits rows of DataFrame d randomly into a and b"""
        n = d.shape[0]
        idx = np.arange(n)
        idx_b = self.rng.choice(idx, int(size_b * n), replace=False)
        idx_b = np.in1d(idx, idx_b)
        a = d.loc[~idx_b]
        b = d.loc[idx_b]
        return (a, b)

    def process(self, path):
        sample = pt.splitext(pt.basename(path))[0]
        d = data.read_cpg(path, self.chromo, self.nrows)
        if self.pos_min is not None:
            d = d.loc[d.pos >= self.pos_min]
        if self.pos_max is not None:
            d = d.loc[d.pos <= self.pos_max]
        test, t = self.split(d, self.test_size)
        train, val = self.split(t, self.val_size)
        sets = {'train': train, 'test': test, 'val': val}
        for k, v in sets.items():
            self.write(v, k, sample)
