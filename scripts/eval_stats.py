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


def get_fun(name):
    return eval_stats.__dict__[name]


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
                       help='Default window length',
                       default=3000,
                       type=int)
        p.add_argument('--wlen_stat',
                       help='Window length of specific statistics [stat=wlen]',
                       nargs='+')
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

        stats_wlen = dict()
        for stat in opts.stats:
            stats_wlen[stat] = opts.wlen
        if opts.wlen_stat is not None:
            for wlen_stat in opts.wlen_stat:
                stat, wlen = wlen_stat.split('=')
                wlen = int(wlen)
                stats_wlen[stat] = wlen

        stats = dict()
        for stat_name in opts.stats:
            if stat_name.find('win') >= 0:
                def tfun(x, name=stat_name, wlen=stats_wlen[stat_name]):
                    return get_fun(name)(x, delta=wlen / 2)
                fun = tfun
            else:
                fun = get_fun(stat_name)
            stats[stat_name] = fun

        log.info('Process ...')
        in_path, in_group = hdf.split_path(opts.in_file)
        if opts.out_file:
            out_path, out_group = hdf.split_path(opts.out_file)
        else:
            out_path = in_path
            out_group = '/es'
        p = eval_stats.Processor(in_path)
        p.in_group = in_group
        p.out_path = out_path
        p.out_group = out_group
        p.chromos = opts.chromos
        p.logger = lambda x: log.info(x)
        p.process(stats)

        log.info('Done!')
        return 0


if __name__ == '__main__':
    app = EvalStats()
    app.run(sys.argv)
