#!/usr/bin/env python

import argparse
import sys
import logging
import os
import os.path as pt
import pandas as pd

from predict.data.annos import read_annos_matrix


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
            description='Extract annotation matrix from file')
        p.add_argument(
            'pos_file',
            help='Position file with chromo and pos columns')
        p.add_argument(
            '-o', '--out_file',
            help='Output file',
            default='annos.h5')
        p.add_argument(
            '--annos_file',
            help='Annotation file',
            default=os.getenv('Cannos'))
        p.add_argument(
            '--annos',
            help='Regex of annotations to be considered',
            default=[r'^loc.+$'],
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

        self.log = log
        self.opts = opts

        pos = pd.read_table(opts.pos_file, dtype={0: str}, header=None)
        pos.columns = ['chromo', 'pos']
        log.info('Read annotations')
        for chromo in pos.chromo.unique():
            log.info('Chromosome %s' % (chromo))
            cpos = pos.loc[pos.chromo == chromo]
            annos = read_annos_matrix(opts.annos_file, chromo=chromo,
                                      pos=cpos.pos.values,
                                      regexs=opts.annos)
            annos.to_hdf(opts.out_file, chromo)

        log.info('Done!')

        return 0

if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
