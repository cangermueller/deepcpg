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
            description='Description')
        p.add_argument(
            'in_files',
            help='HDF files',
            nargs='+')
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

        pos = dict()
        for in_file in opts.in_files:
            f = h5.File(in_file, 'r')
            g = f['/cpg']
            chromos = list(g.keys())
            for chromo in chromos:
                if chromo not in pos:
                    pos[chromo] = set()
                p = g['%s/pos' % (chromo)].value
                pos[chromo].update(p)
            f.close()
        for chromo in sorted(pos.keys()):
            p = sorted(list(pos[chromo]))
            d = pd.DataFrame(dict(chromo=chromo, pos=p))
            s = d.to_csv(opts.out_file, sep='\t', header=False, index=False, mode='a')
            if s is not None:
                print(s, end='')

        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
