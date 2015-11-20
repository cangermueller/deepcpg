#!/usr/bin/env python

import argparse
import sys
import logging
import os
import os.path as pt
import pandas as pd
import numpy as np
import h5py as h5

from utils import evaluate_all, read_data, load_model, MASK
from predict.evaluation import eval_to_str


def write_test(test_file, d):
    f = h5.File(test_file, 'a')
    start = 0
    chromos = [x.decode() for x in d['chromos']]
    for i, chromo in enumerate(chromos):
        end = start + d['chromos_len'][i]
        e = {k: d[k] for k in ['pos', 'y', 'z']}
        for k in e.keys():
            e[k] = e[k][start:end]
        t = e['y'] != MASK
        for k in e.keys():
            e[k] = e[k][t]
        for k, v in e.items():
            k = pt.join(chromo, k)
            if k in f:
                del f[k]
            f.create_dataset(k, data=v)
        start = end
    f.close()

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

        #  z = dict()
        #  for k, v in data.items():
            #  if k.startswith('u'):
                #  z[k] = np.random.binomial(1, 0.5, len(v))

        log.info('Load model')
        model = load_model(opts.model_file, opts.model_weights_file)

        f = h5.File(opts.data_file)
        labels = dict()
        for k in ['label_units', 'label
        label_units = f['label_units'].value
        reader = DataReader(opts.data_file, shuffle=False)
        prev_chromo = ''
        z = None
        g = None
        for chromo, i, j in reader:
            if chromo != prev_chromo:
                if z is not None:
                    pass
                g = f[chromo]
            d = {k: g[k][i:j] for k in g.keys()}
            z = model.predict(d)






        log.info('Predict')
        z = model.predict(data)

        log.info ('Evaluate')
        p = evaluate_all(data, z)

        print('Performance:')
        print(eval_to_str(p))

        log.info('Write')
        labels = dict()
        for a, b in zip(data['label_units'], data['label_files']):
            labels[a.decode()] = b.decode()
        for t in z.keys():
            log.info(labels[t])
            d = {k: data[k] for k in ['chromos', 'chromos_len', 'pos']}
            d['z'] = z[t]
            d['y'] = data[t]
            out_dir = pt.join(opts.out_dir, labels[t])
            os.makedirs(out_dir, exist_ok=True)
            write_test(pt.join(out_dir, 'test.h5'), d)
        log.info('Done!')

        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
