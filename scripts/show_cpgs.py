#!/usr/bin/env python

import argparse
import sys
import logging
import os
import os.path as pt
import h5py as h5
import pandas as pd


__dir = pt.dirname(pt.realpath(__file__))


def get_cpg(path, chromo, start, end):
    d = pd.read_table(path, header=None, dtype={0: 'str'}, usecols=[0, 1, 2])
    d.columns=['chromo', 'pos', 'value']
    d = d.query('chromo==@chromo and pos >= @start and pos <= @end')
    return d


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
            description='Print CpGs of sample')
        p.add_argument(
            'chromo',
            help='Chromosome')
        p.add_argument(
            'start',
            help='Start position or window center (offset=1)',
            type=int)
        p.add_argument(
            'end',
            help='End position or window delta (offset=1)',
            type=int)
        p.add_argument(
            'sample',
            help='Sample name')
        p.add_argument(
            '-d', '--delta',
            help='Show sequence centered on position',
            action='store_true')
        p.add_argument(
            '--dataset',
            help='Name of dataset',
            default='scBS14')
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

        if pt.isdir(opts.dataset):
            ddir = opts.dataset
        else:
            ddir = pt.join(pt.join(os.getenv('Pdata'), opts.dataset))

        sample = pt.splitext(pt.basename(opts.sample))[0]
        sample += '.bed'
        path = pt.join(ddir, sample)

        chromo = str(opts.chromo)

        if opts.delta:
            pos = opts.start
            start = pos - opts.end
            end = pos + opts.end
        else:
            start = opts.start
            end = opts.end

        d = get_cpg(path, chromo, start, end)
        s = d.to_csv(None, sep='\t', index=False)
        print(s, end='')

        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
