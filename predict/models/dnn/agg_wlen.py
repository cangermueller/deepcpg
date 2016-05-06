#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import h5py as h5

import predict.utils as ut


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
            description='Aggregates values')
        p.add_argument(
            'in_file',
            help='Input file')
        p.add_argument(
            '-o', '--out_file',
            help='Output file',
            default='merged.h5')
        p.add_argument(
            '--axis',
            help='Aggregation axis',
            type=int,
            default=1)
        p.add_argument(
            '--fun',
            help='Aggregation function',
            choices=['mean', 'wmean', 'max'])
        p.add_argument(
            '--wlen',
            help='Slice wlen at center',
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

        in_file = h5.File(opts.in_file, 'r')
        out_file = h5.File(opts.out_file, 'w')
        for k, v in in_file.items():
            log.info(k)
            v = v.value
            if v.ndim > 1:
                v = ut.aggregate(v, fun=opts.fun, wlen=opts.wlen)
            out_file[k] = v
        log.info('Done!')
        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
