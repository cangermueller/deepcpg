#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import pandas as pd
import warnings

__dir = pt.dirname(pt.realpath(__file__))
# sys.path.insert(0, pt.join(__dir, '../module'))

import hdf
import data


def add_pos(path, dataset):
    chromos = hdf.ls(path, pt.join(dataset, 'cpg'))
    for chromo in chromos:
        p = data.get_pos(path, dataset, chromo)
        p = pd.Series(p)
        group = pt.join(dataset, 'pos', chromo)
        p.to_hdf(path, group, format='t', data_columns=True)


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
            description='Adds position vector')
        p.add_argument(
            'in_file',
            help='Input HDF path to dataset (test, train, val)')
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

        log.info('Add position vectors ...')
        hdf_path, hdf_group = hdf.split_path(opts.in_file)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            add_pos(hdf_path, hdf_group)
        log.info('Done!')
        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
