#!/usr/bin/env python

import argparse
import sys
import logging
import os
import os.path as pt
import h5py as h5
import pandas as pd
from glob import glob


__dir = pt.dirname(pt.realpath(__file__))


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
            description='Show samples of dataset')
        p.add_argument(
            '--dataset',
            help='Name of dataset',
            default='scBS14')
        p.add_argument(
            '--path',
            help='Show full path name',
            action='store_true')
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
            ddir = pt.join(pt.join(os.getenv('Bdata'), opts.dataset))

        s = glob(pt.join(ddir, '*.bed'))
        if not opts.path:
            s = [pt.splitext(pt.basename(x))[0] for x in s]
        s = sorted(s)
        for x in s:
            print(x)

        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
