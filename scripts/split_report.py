#!/usr/bin/env python

import argparse
import sys
import logging
import pandas as pd
import numpy as np
import os.path as pt


def read_report(path, samples=None, nrows=None):
    if samples is None:
        header = pd.read_table(path, nrows=1)
        columns = list(header.columns)
        start_idx = columns.index('Distance') + 1
        samples = columns[start_idx:]
    columns = ['Chromosome', 'Start'] + samples
    d = pd.read_table(path, usecols=columns,
                      dtype=dict(Chromosome=np.str, Start=np.int, End=np.int),
                      nrows=nrows)
    d.rename(columns={'Chromosome': 'chromo', 'Start': 'pos'},
             inplace=True)
    d = d.set_index(['chromo', 'pos'])
    d /= 100
    return d


def to_bed(d, dirname):
    samples = d.columns
    d = d.reset_index()
    for sample in samples:
        ds = d[['chromo', 'pos', sample]]
        ds = ds.dropna()
        path = pt.join(dirname, sample + '.bed')
        ds.to_csv(path, sep='\t', index=False, header=False)


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
            description='Splits Seqmonk report with methylation rates')
        p.add_argument(
            'in_file',
            help='Seqmonk report with methylation rates.')
        p.add_argument(
            '-o', '--out_dir',
            help='Output directory',
            default='.')
        p.add_argument(
            '--samples',
            help='Only extract these samples')
        p.add_argument(
            '--nrows',
            help='Only read that many rows',
            type=int)
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

        log.info('Read report ...')
        d = read_report(opts.in_file, opts.samples, opts.nrows)

        log.info('Split report ...')
        to_bed(d, opts.out_dir)

        log.info('Done!')
        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
