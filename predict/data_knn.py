#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import pandas as pd
import warnings

__dir = pt.dirname(pt.realpath(__file__))
# sys.path.insert(0, pt.join(__dir, '../module'))

import data
import hdf
import feature_extractor as fext


class Processor(object):

    def __init__(self, knn_ext):
        self.knn_ext = knn_ext
        self.out_path = None
        self.chromos = None
        self.samples = None
        self.nrows = None

    def process_sample(self, d, sample):
        pos = d.index
        cpg = d[sample].dropna()
        f = self.knn_ext.extract(pos, cpg.index, cpg.values)
        f = pd.DataFrame(f, columns=self.knn_ext.labels)
        f['pos'] = pos
        f = pd.melt(f, id_vars='pos', var_name='feature',
                    value_name='value')
        return f

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

    def process_chromo(self, path, dataset, chromo):
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


class App(object):

    def run(self, args):
        name = pt.basename(args[0])
        parser = self.create_parser(name)
        opts = parser.parse_args(args[1:])
        return self.main(name, opts)

    def create_parser(self, name):
        p = argparse.ArgumentParser(
            prog=name,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Computes KNN features')
        p.add_argument(
            'in_file',
            help='Input HDF path to dataset (test, train, val)')
        p.add_argument(
            '-o', '--out_file',
            help='Output HDF path if different from input HDF path')
        p.add_argument(
            '-k', '--knn',
            help='Number of k nearest neighbors',
            type=int,
            default=5)
        p.add_argument(
            '--chromos',
            help='Only apply to these chromosome',
            nargs='+')
        p.add_argument(
            '--samples',
            help='Only apply to these samples',
            nargs='+')
        p.add_argument(
            '--nrows',
            help='Only read that many CpG sites from each sample',
            type=int)
        p.add_argument(
            '--verbose', help='More detailed log messages', action='store_true')
        p.add_argument(
            '--log_file', help='Write log messages to file')
        return p

    def main(self, name, opts):
        logging.basicConfig(filename=opts.log_file,
                            format='%(levelname)s (%(asctime)s): %(message)s')
        log = logging.getLogger(name)
        if opts.verbose:
            log.setLevel(logging.DEBUG)
        else:
            log.setLevel(logging.INFO)
        log.debug(opts)

        in_path, in_group = hdf.split_path(opts.in_file)
        if opts.out_file:
            out_path, out_group = hdf.split_path(opts.out_file)
        else:
            out_path = in_path
            out_group = in_group
        fe = fext.KnnCpgFeatureExtractor(opts.knn)
        p = Processor(fe)
        p.out_path = out_path
        p.out_group = out_group
        p.chromos = opts.chromos
        p.samples = opts.samples
        p.nrows = opts.nrows

        log.info('Process ...')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            p.process(in_path, in_group)
        log.info('Done!')
        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
