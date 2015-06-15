#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import pandas as pd
import warnings

from predict import hdf
from predict import feature_extractor as fext
from predict import data
from predict import annos as A
from predict import data_annos


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
            description='Adds annotation')
        p.add_argument(
            'in_file',
            help='Input HDF path to dataset (test, train, val)')
        p.add_argument(
            '-a', '--anno_files',
            help='Annotation files in BED format',
            nargs='+')
        p.add_argument(
            '--prefix',
            help='Prefix annotation names',
            default='')
        p.add_argument(
            '--distance',
            help='Compute distance to annotations',
            action='store_true'
        )
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

        log.info('Add annotations ...')
        in_path, in_group = hdf.split_path(opts.in_file)
        p = data_annos.Processor(in_path, in_group, opts.distance)
        for anno_file in opts.anno_files:
            anno_name = opts.prefix + pt.splitext(pt.basename(anno_file))[0]
            log.info('\t%s...', anno_name)
            annos = data.read_annos(anno_file)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                p.process(annos, anno_name)
        log.info('Done!')
        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
