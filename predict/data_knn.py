import os.path as pt
import numpy as np
import pandas as pd

from predict import data
from predict import hdf


class Processor(object):

    def __init__(self, in_path, knn_ext):
        self.knn_ext = knn_ext
        self.in_path = in_path
        self.in_group = None
        self.out_path = None
        self.out_group = None
        self.chromos = None
        self.samples = None
        self.logger = None

    def log(self, msg):
        if self.logger is not None:
            self.logger(msg)

    def write(self, f, chromo, sample):
        t = f.feature.str.contains('dist_')
        k = self.knn_ext.k
        out_group = pt.join(self.out_group, 'knn%d' % (k), chromo, sample)
        f[~t].to_hdf(self.out_path, out_group)
        if np.sum(t) > 0:
            out_group = pt.join(self.out_group,
                                'knn%d_dist' % (k), chromo, sample)
            f[t].to_hdf(self.out_path, out_group)

    def process_sample(self, chromo, sample, pos):
        self.log('Sample %s ...' % (sample))

        def read_cpgs(group):
            t = pt.join(self.in_group, group, 'cpg', chromo, sample)
            cpg = pd.read_hdf(self.in_path, t)
            return cpg

        # knn wrt. training CpGs, excluding test CpGs!
        cpg = read_cpgs('train')
        cpg = pd.concat([cpg, read_cpgs('val')])
        assert len(np.unique(cpg.index.values)) == len(cpg)
        cpg = cpg.sort_index()
        f = self.knn_ext.extract(pos, cpg.index.values, cpg.value.values)
        f = pd.DataFrame(f, columns=self.knn_ext.labels)
        f['pos'] = pos
        f = pd.melt(f, id_vars='pos', var_name='feature',
                    value_name='value')
        return f

    def process_chromo(self, chromo):
        self.log('Chromosome %s ...' % (str(chromo)))
        pos = data.read_pos(self.in_path, self.in_group, chromo)
        if self.samples:
            samples = self.samples
        else:
            samples = hdf.ls(self.in_path,
                             pt.join(self.in_group, 'train', 'cpg', chromo))
        for sample in samples:
            f = self.process_sample(chromo, sample, pos)
            self.write(f, chromo, sample)

    def process(self):
        if self.out_path is None:
            self.out_path = self.in_path
        if self.out_group is None:
            self.out_group = self.in_group
        if self.chromos:
            chromos = self.chromos
        else:
            chromos = hdf.ls(self.in_path, pt.join(self.in_group, 'pos'))
        for chromo in chromos:
            self.process_chromo(chromo)
