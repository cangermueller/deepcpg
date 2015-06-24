#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import pandas as pd
import warnings

from predict import data
from predict import hdf
from predict import feature_extractor as fext
from predict import data_knn


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
            '--no_dist',
            help='Do not compute distance to knn',
            action='store_true')
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
        fe = fext.KnnCpgFeatureExtractor(opts.knn, dist=not opts.no_dist)
        p = data_knn.Processor(fe)
        p.out_path = out_path
        p.out_group = out_group
        p.chromos = opts.chromos
        p.samples = opts.samples
        p.nrows = opts.nrows
        p.logger = lambda x: log.info(x)

        log.info('Process ...')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            p.process(in_path, in_group)
        log.info('Done!')
        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
