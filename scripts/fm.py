#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import warnings
import pandas as pd

from predict import hdf
from predict import data_select as dsel


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
            description='Builds feature matrix')
        p.add_argument(
            'in_file',
            help='HDF path to dataset')
        p.add_argument(
            '-o', '--out_file',
            help='Output HDF path')
        p.add_argument(
            '--kmers',
            help='HDF path to kmers')
        p.add_argument(
            '--verbose',
            help='More detailed log messages',
            action='store_true')
        p.add_argument(
            '--log_file',
            help='Write log messages to file')
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

        fs = dsel.FeatureSelection()
        fs.knn = True
        fs.knn_dist = False
        fs.annos = True

        sel = dsel.Selector(fs)
        sel.samples = None
        chromos = [1]

        in_path, in_group = hdf.split_path(opts.in_file)
        for chromo in chromos:
            log.info('Chromosome %d' % chromo)

            log.info('Select ...')
            sel.chromos = [chromo]
            D = sel.select(in_path, in_group)
            D.index = D.index.droplevel(0)

            if opts.kmers is not None:
                path, group = hdf.split_path(opts.kmers)
                kmers = pd.read_hdf(path, pt.join(group, str(chromo)))
                t = pd.MultiIndex.from_product([['kmers'], kmers.columns])
                kmers.columns = t
                t = D.shape[0]
                D = pd.concat([D, kmers], axis=1, join='inner')
                assert D.shape[0] == t

            log.info('Format ...')
            Y = D.loc[:, 'cpg']
            X = D.loc[:, D.columns.get_level_values(0) != 'cpg']
            t = X.notnull().all(axis=1)
            Y = Y.loc[t]
            X = X.loc[t]
            assert X.isnull().sum().sum() == 0

            log.info('Write ...')
            path, group = hdf.split_path(opts.out_file)

            def write(D, name):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    D.to_hdf(path, pt.join(group, name, str(chromo)))

            write(X, 'X')
            write(Y, 'Y')

        log.info('Done!')
        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
