#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import numpy as np
import pandas as pd
import h5py as h5

from predict import feature_extractor as fext


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
            description='Extract k neighboring CpG sites')
        p.add_argument(
            'in_file',
            help='data file')
        p.add_argument(
            '--out_file',
            help='Output HDF file if different from input')
        p.add_argument(
            '--out_group',
            help='Output group in HDF file',
            default='knn')
        p.add_argument(
            '--pos_file',
            help='Position vector')
        p.add_argument(
            '-k', '--knn',
            help='Number of k nearest neighbors',
            type=int,
            default=5)
        p.add_argument(
            '--no_dist',
            help='Do not compute distance to knn',
            action='store_true')
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

        in_file = h5.File(opts.in_file, 'a')
        chromos = opts.chromos
        if chromos is None:
            chromos = list(in_file['cpg'].keys())

        if opts.out_file:
            out_file = h5.File(opts.out_file, 'a')
        else:
            out_file = in_file

        knn_pos = None
        if opts.pos_file is not None:
            knn_pos = pd.read_table(opts.pos_file, header=None, dtype={0: 'str', 1: 'int32'})
            knn_pos.columns=['chromo', 'pos']
            knn_pos.set_index('chromo', inplace=True)

        fe = fext.KnnCpgFeatureExtractor(opts.knn, dist=not opts.no_dist)
        dist_cols = np.array([x.startswith('dist') for x in fe.labels])

        for chromo in chromos:
            log.info('Chromosome %s' % (chromo))
            def read(name):
                return in_file['/cpg/%s/%s' % (chromo, name)].value

            cpos = read('pos')
            ccpg = read('cpg')

            if knn_pos is None:
                cknn_pos = cpos
            else:
                cknn_pos = knn_pos.loc[chromo, 'pos'].values
            cknn_pos.sort()
            f = fe.extract(cknn_pos, cpos, ccpg)

            def write(name, data):
                g = '/%s/%s/%s' % (opts.out_group, chromo, name)
                if g in out_file:
                    del out_file[g]
                out_file.create_dataset(g, data=data, compression='gzip')

            cknn_pos = cknn_pos.astype('int32', copy=False)
            write('pos', cknn_pos)
            cknn = f[:, ~dist_cols]
            cknn[np.isnan(cknn)] = 0
            cknn = cknn.astype('int8', copy=False)
            write('knn', cknn)
            if len(dist_cols):
                cdist = f[:, dist_cols]
                cdist[np.isnan(cdist)] = np.iinfo('int32').max - 100
                cdist = cdist.astype('int32', copy=False)
                write('dist', cdist)
        in_file.close()
        if out_file is not in_file:
            out_file.close()
        log.info('Done!')
        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
