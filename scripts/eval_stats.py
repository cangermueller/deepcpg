#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import numpy as np
import pandas as pd
import warnings

from predict import utils as ut
from predict import hdf
from predict import eval_stats


class EvalStats(object):

    def run(self, args):
        name = pt.basename(args[0])
        parser = self.create_parser(name)
        opts = parser.parse_args(args[1:])
        return self.main(name, opts)

    def create_parser(self, name):
        p = argparse.ArgumentParser(
            prog=name,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Computes evaluation statistics')
        p.add_argument(
            'in_file',
            help='HDF path of data set')
        p.add_argument('-o', '--out_file',
                       help='HDF path of output statistics')
        p.add_argument('--stats',
                       help='Name of evaluation functions',
                       nargs='+')
        p.add_argument('--wlen',
                       help='Window length for windowing functions',
                       default=3000,
                       type=int)
        p.add_argument('--chromos',
                       help='Only consider these chromosomes',
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

        stats = dict()
        for stat_name in opts.stats:
            if stat_name.find('win') >= 0:
                def tfun(x, name=stat_name):
                    return globals()[name](x, delta=opts.wlen / 2)
                fun = tfun
            else:
                fun = globals()[stat_name]
            stats[stat_name] = fun

        log.info('Process ...')
        p = eval_stats.Processor(opts.out_file)
        p.chromos = opts.chromos
        p.process(opts.in_file, stats)

        log.info('Done!')
        return 0


if __name__ == '__main__':
    app = EvalStats()
    app.run(sys.argv)
