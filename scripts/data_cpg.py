#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import numpy as np
import warnings

from predict import hdf
from predict import data
from predict import data_cpg


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
            help='Output HDF path.',
            default='data.h5')
        p.add_argument(
            '--chromos',
            help='Only use these chromosomes',
            nargs='+')
        p.add_argument(
            '--nrows',
            help='Only read that many rows from each file',
            type=int)
        p.add_argument(
            '--start',
            help='Start position on chromosome',
            type=int)
        p.add_argument(
            '--stop',
            help='Stop position on chromosome',
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

        hdf_path, hdf_group = hdf.split_path(opts.out_file)
        p = data_cpg.Processor(hdf_path, hdf_group)
        p.chromos = opts.chromos
        p.nrows = opts.nrows
        p.pos_min = opts.start
        p.pos_max = opts.stop

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
