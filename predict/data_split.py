import numpy as np
import os.path as pt
import pandas as pd

from predict import hdf


class Processor(object):

    def __init__(self, in_path):
        self.in_path = in_path
        self.out_path = in_path
        self.in_group = '/'
        self.out_group = self.in_group
        self.test_size = 0.3
        self.val_size = 0.1
        self.rng = np.random.RandomState(0)
        self.logger = None

    def log(self, x):
        if self.logger is not None:
            self.logger(x)

    def write(self, d, name, chromo, sample):
        t = pt.join(self.out_group, name, 'cpg', chromo, sample)
        d.to_hdf(self.out_path, t)


    def split(self, d, size_b=0.5):
        """Splits rows of DataFrame d randomly into a and b"""
        n = d.shape[0]
        idx = np.arange(n)
        idx_b = self.rng.choice(idx, int(size_b * n), replace=False)
        idx_b = np.in1d(idx, idx_b)
        a = d.loc[~idx_b]
        b = d.loc[idx_b]
        return (a, b)


    def process_chromo(self, chromo):
        g = pt.join(self.in_group, 'cpg', chromo)
        samples = hdf.ls(self.in_path, g)
        for sample in samples:
            self.log('%s ...' % (sample))
            a = pd.read_hdf(self.in_path, pt.join(g, sample))
            a, b = self.split(a, self.test_size)
            self.write(b, 'test', chromo, sample)
            a, b = self.split(a, self.val_size)
            self.write(b, 'val', chromo, sample)
            self.write(a, 'train', chromo, sample)

    def process(self):
        chromos = hdf.ls(self.in_path, pt.join(self.in_group, 'cpg'))
        for chromo in chromos:
            self.log('Chromosome %s ...' % (chromo))
            self.process_chromo(chromo)
