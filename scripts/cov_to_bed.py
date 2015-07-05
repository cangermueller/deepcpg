#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import pandas as pd
import numpy as np


__dir = pt.dirname(pt.realpath(__file__))
# sys.path.insert(0, pt.join(__dir, '../module'))


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
            description='Converts bismark coverage file to bed.')
        p.add_argument(
            'in_file',
            help='Input coverage file',
            nargs='?')
        p.add_argument(
            '-o', '--out_file',
            help='Output file')
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

        in_file = opts.in_file if opts.in_file else sys.stdin
        d = pd.read_table(in_file, header=None, usecols=[0, 1, 4, 5],
                          dtype={0: np.str})
        d.columns = ['chromo', 'pos', 'npos', 'nneg']
        d['rate'] = d.npos / (d.npos + d.nneg)
        d = d.loc[:, ['chromo', 'pos', 'rate']]
        s = d.to_csv(opts.out_file, sep='\t', header=None, index=False)
        if s is not None:
            print(s, end='')

        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
