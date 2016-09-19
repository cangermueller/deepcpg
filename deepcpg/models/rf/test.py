#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import h5py as h5
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from predict.evaluation import evaluate, eval_to_str


def read_data(path, max_samples=None):
    f = h5.File(path)
    d = dict()
    for k in ['X', 'y', 'pos', 'chromos', 'chromos_len']:
        g = f[k]
        if max_samples is None:
            d[k] = g.value
        else:
            d[k] = g[:max_samples]
    f.close()
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
            description='Predict output')
        p.add_argument(
            'data_file',
            help='HDF path to data')
        p.add_argument(
            'model_file',
            help='Path to model')
        p.add_argument(
            '-o', '--out_file',
            help='Output file',
            default='predict.h5')
        p.add_argument(
            '--num_cores',
            help='Number of CPU cores',
            type=int,
            default=1)
        p.add_argument(
            '--max_samples',
            help='Limit # samples',
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

    def main(self, name, opts):
        logging.basicConfig(filename=opts.log_file,
                            format='%(levelname)s (%(asctime)s): %(message)s')
        log = logging.getLogger(name)
        if opts.verbose:
            log.setLevel(logging.DEBUG)
        else:
            log.setLevel(logging.INFO)
            log.debug(opts)

        log.info('Read model')
        with open(opts.model_file, 'rb') as f:
            m = pickle.load(f)
        print(m)
        print()

        log.info('Read data')
        d = read_data(opts.data_file, opts.max_samples)
        print('Data: %d x %d' % d['X'].shape)

        z = m.predict_proba(d['X'])[:, 1]
        log.info('Write')
        f = h5.File(opts.out_file, 'a')
        start = 0
        for i, chromo in enumerate([x.decode() for x in d['chromos']]):
            end = start + d['chromos_len'][i]
            e = dict(pos=d['pos'], y=d['y'], z=z)
            for k, v in e.items():
                k = pt.join(chromo, k)
                if k in f:
                    del f[k]
                f.create_dataset(k, data=v[start:end])
            start = end
        f.close()

        p = evaluate(d['y'], z)
        print('Performance:')
        print(eval_to_str(p))

        log.info('Done!')

        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
