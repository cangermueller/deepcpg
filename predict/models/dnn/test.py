#!/usr/bin/env python

import argparse
import sys
import logging
import os
import os.path as pt
import pandas as pd
import numpy as np
import h5py as h5

from utils import DataReader, read_chromos, read_labels, read_and_stack
from utils import MASK, load_model


class App(object):

    def run(self, args):
        name = pt.basename(args[0])
        parser = self.create_parser(name)
        opts = parser.parse_args(args[1:])
        self.opts = opts
        return self.main(name, opts)

    def create_parser(self, name):
        p = argparse.ArgumentParser(
            prog=name,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Make prediction on data set')
        p.add_argument(
            'data_file',
            help='Data file')
        p.add_argument(
            'model_file',
            help='Model json file')
        p.add_argument(
            'model_weights_file',
            help='Model weights file')
        p.add_argument(
            '-o', '--out_dir',
            help='Output directory')
        p.add_argument(
            '--chunk_size',
            help='Size of training chunks',
            type=int,
            default=10**7)
        p.add_argument(
            '--max_chunks',
            help='Limit # training chunks',
            type=int)
        p.add_argument(
            '--seed',
            help='Seed of rng',
            type=int,
            default=0)
        p.add_argument(
            '--verbose',
            help='More detailed log messages',
            action='store_true')
        p.add_argument(
            '--log_file',
            help='Write log messages to file')
        return p

    def write_z(self, z, chromo, labels):
        lmap = dict()
        for target, _file in zip(labels['targets'], labels['files']):
            lmap[target] = _file
        for t in z['z'].keys():
            p = pt.join(self.opts.out_dir, lmap[t])
            os.makedirs(p, exist_ok=True)
            f = h5.File(pt.join(p, 'test.h5'), 'a')
            if chromo in f:
                del f[chromo]
            g = f.create_group(chromo)
            for k in z.keys():
                g[k] = z[k][t]
            f.close()

    def main(self, name, opts):
        logging.basicConfig(filename=opts.log_file,
                            format='%(levelname)s (%(asctime)s): %(message)s')
        log = logging.getLogger(name)
        if opts.verbose:
            log.setLevel(logging.DEBUG)
        else:
            log.setLevel(logging.INFO)
            log.debug(opts)

        if opts.seed is not None:
            np.random.seed(opts.seed)
        pd.set_option('display.width', 150)

        log.info('Load model')
        model = load_model(opts.model_file, opts.model_weights_file)

        def read_predict(chromo, i, j):
            f = h5.File(opts.data_file)
            g = f[chromo]
            d = {k: g[k][i:j] for k in g.keys()}
            f.close()
            r = dict()
            #  r['y'] = {k: d[k] for k in ['u0_y', 'u1_y', 'u2_y']}
            #  r['z'] = {k: np.random.uniform(0, 1, len(r['y'][k])) for k in r['y'].keys()}
            r['z'] = model.predict(d)
            for k in r['z'].keys():
                r['z'][k] = np.ravel(r['z'][k])
            r['y'] = {k: d[k] for k in r['z'].keys()}
            r['pos'] = {k: d['pos'] for k in r['z'].keys()}
            for k in r['z'].keys():
                t = r['y'][k] != MASK
                for x in r.keys():
                    r[x][k] = r[x][k][t]
            return r

        labels = read_labels(opts.data_file)
        chromos = read_chromos(opts.data_file)
        for chromo in chromos:
            log.info('Chromosome %s' % chromo)
            reader = DataReader(opts.data_file, chunk_size=opts.chunk_size,
                                chromos=[chromo],
                                max_chunks=opts.max_chunks)
            z = read_and_stack(reader, read_predict)
            self.write_z(z, chromo, labels)

        log.info('Done!')

        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
