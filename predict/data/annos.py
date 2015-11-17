#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import numpy as np
import pandas as pd
import h5py as h5

from predict import annos as A


def annotate(pos, annos):
    pos = pos.sort().pos.values
    annos = annos.sort()
    start, end = A.join_overlapping(annos.start.values,
                                    annos.end.values)
    x = A.in_which(pos, start, end)
    return x


def read_annos(path):
    d = pd.read_table(path, header=None, usecols=[0, 1, 2],
                      dtype={0: 'str', 1: 'int32', 2: 'int32'})
    d.columns = ['chromo', 'start', 'end']
    d.chromo = d.chromo.str.lower().str.replace('chr', '')
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
            description='Annotate position vector')
        p.add_argument(
            'pos_file',
            help='Positions')
        p.add_argument(
            'anno_files',
            help='Annotations in BED format',
            nargs='+')
        p.add_argument(
            '-o', '--out_file',
            help='Output HDF file')
        p.add_argument(
            '--chromos',
            help='Only use these chromosomes',
            nargs='+')
        p.add_argument(
            '--verbose', help='More detailed log messages', action='store_true')
        p.add_argument(
            '--log_file', help='Write log messages to file')
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

        log.info('Read positions')
        pos = pd.read_table(opts.pos_file, header=None, dtype={0: 'str', 1: 'int32'})
        pos.columns=['chromo', 'pos']
        if opts.chromos is not None:
            pos = pos.loc[pos.chromo.isin(opts.chromos)]

        chromos = pos.chromo.unique()

        f = h5.File(opts.out_file, 'a')
        for anno_file in opts.anno_files:
            anno = pt.splitext(pt.basename(anno_file))[0]
            log.info(anno)
            annos = read_annos(anno_file)
            for chromo in chromos:
                cpos = pos.loc[pos.chromo == chromo]
                cannos = annos.loc[annos.chromo == chromo]
                y = annotate(cpos, cannos)
                t = pt.join(chromo, anno)
                if t in f:
                    del f[t]
                g = f.create_group(t)
                g['pos'] = cpos.pos.values
                g['annos'] = y
        f.close()

        log.info('Done!')
        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
