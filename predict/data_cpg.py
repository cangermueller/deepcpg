#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import numpy as np
import warnings

__dir = pt.dirname(pt.realpath(__file__))
# sys.path.insert(0, pt.join(__dir, '../module'))

import hdf
import data


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
            description='Reads CpG methylation rates [0;1] from bed files and \
            split data into training test and validation set')
        p.add_argument(
            'in_files',
            help='Input files with methylation rates in bed format \
            (chromo pos rate).',
            nargs='+')
        p.add_argument(
            '-o', '--out_file',
            help='Output HDF path. Creates train, test, val groups.',
            default='data.h5')
        p.add_argument(
            '--test_size',
            help='Size of test set',
            type=float,
            default=0.5)
        p.add_argument(
            '--val_size',
            help='Size of validation set',
            type=float,
            default=0.1)
        p.add_argument(
            '--chromo',
            help='Only process data from single chromosome')
        p.add_argument(
            '--nrows',
            help='Only read that many rows from each file',
            type=int)
        p.add_argument(
            '--start',
            help='Start position on chromomse',
            type=int)
        p.add_argument(
            '--stop',
            help='Stop position on chromosome',
            type=int)
        p.add_argument(
            '--seed',
            help='Seed of rng',
            type=int,
            default=0)
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

        hdf_path, hdf_group = hdf.split_path(opts.out_file)
        p = Processor(hdf_path, hdf_group, opts.test_size, opts.val_size)
        p.chromo = opts.chromo
        p.nrows = opts.nrows
        p.pos_min = opts.start
        p.pos_max = opts.stop
        p.rng = np.random.RandomState(opts.seed)

        log.info('Process files ...')
        for in_file in opts.in_files:
            log.info('\t%s', in_file)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                p.process(in_file)

        log.info('Done!')
        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
