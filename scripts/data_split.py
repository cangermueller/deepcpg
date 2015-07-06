#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import numpy as np
import warnings

from predict import hdf
from predict import data
from predict import data_split


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
            description='Split CpGs into train, test, and validation set')
        p.add_argument(
            'in_file',
            help='Input HDF path of data file.')
        p.add_argument(
            '--test_size',
            help='Size of test set',
            type=float,
            default=0.3)
        p.add_argument(
            '--val_size',
            help='Size of validation set',
            type=float,
            default=0.1)
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

        in_path, in_group = hdf.split_path(opts.in_file)
        p = data_split.Processor(in_path)
        p.test_size = opts.test_size
        p.val_size = opts.val_size
        p.rng = np.random.RandomState(opts.seed)
        p.logger = lambda x: log.info(x)

        log.info('Split ...')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            p.process()
        log.info('Done!')
        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)

