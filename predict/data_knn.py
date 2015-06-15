import argparse
import sys
import logging
import os.path as pt
import pandas as pd
import warnings

from predict import data
from predict import hdf
from predict import feature_extractor as fext


class Processor(object):

    def __init__(self, knn_ext):
        self.knn_ext = knn_ext
        self.out_path = None
        self.chromos = None
        self.samples = None
        self.nrows = None
        self.logger = None

    def log(self, msg):
        if self.logger is not None:
            self.logger(msg)

    def write(self, f, path, dataset, chromo, sample):
        if self.out_path:
            out_path = self.out_path
        else:
            out_path = path

        def to_hdf(d, out_group):
            d.to_hdf(out_path, out_group, format='t', data_columns=True)

        t = f.feature.str.contains('dist_')
        k = self.knn_ext.k
        out_group = pt.join(dataset, 'knn%d' % (k), chromo, sample)
        to_hdf(f[~t], out_group)
        out_group = pt.join(dataset, 'knn%d_dist' % (k), chromo, sample)
        to_hdf(f[t], out_group)

    def process_sample(self, d, sample):
        self.log('Sample %s ...' % (sample))
        pos = d.index
        cpg = d[sample].dropna()
        f = self.knn_ext.extract(pos, cpg.index, cpg.values)
        f = pd.DataFrame(f, columns=self.knn_ext.labels)
        f['pos'] = pos
        f = pd.melt(f, id_vars='pos', var_name='feature',
                    value_name='value')
        return f

    def process_chromo(self, path, dataset, chromo):
        self.log('Chromosome %s ...' % (str(chromo)))
        d = data.read_cpg_list(path, dataset, chromo, nrows=self.nrows)
        d = pd.pivot_table(d, index='pos', columns='sample', values='value')
        d = d.sort_index()
        if self.samples:
            samples = self.samples
        else:
            samples = d.columns
        for sample in samples:
            f = self.process_sample(d, sample)
            self.write(f, path, dataset, chromo, sample)

    def process(self, path, dataset):
        if self.chromos:
            chromos = self.chromos
        else:
            chromos = hdf.ls(path, pt.join(dataset, 'cpg'))
        for chromo in chromos:
            self.process_chromo(path, dataset, chromo)
