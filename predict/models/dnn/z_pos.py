#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import h5py as h5
import pandas as pd


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
            description='Write positions from prediction file')
        p.add_argument(
            'in_file',
            help='Prediction file')
        p.add_argument(
            '--target',
            help='Target name')
        p.add_argument(
            '-o', '--out_file',
            help='Output file')
        p.add_argument(
            '--chromos',
            help='Chromosomes',
            nargs='+')
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

        f = h5.File(opts.in_file, 'r')
        target = opts.target
        if target is None:
            target = list(f.keys())[0]
        chromos = opts.chromos
        if chromos is None:
            chromos = list(f[target].keys())
        d = []
        for chromo in chromos:
            log.info(chromo)
            pos = f[pt.join(target, chromo, 'pos')].value
            dc = pd.DataFrame(dict(chromo=chromo, pos=pos))
            d.append(dc)
        d = pd.concat(d)
        s = d.to_csv(opts.out_file, sep='\t', index=False, header=False)
        if s is not None:
            print(s, end='')

        log.info('Done!')

        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
