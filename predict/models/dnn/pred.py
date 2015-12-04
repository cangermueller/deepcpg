#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import pandas as pd
import numpy as np
import h5py as h5

from predict.models.dnn.utils import read_labels, open_hdf, load_model, ArrayView
from utils import MASK
from utils import write_z2 as write_z

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
            '-o', '--out_file',
            help='Output file')
        p.add_argument(
            '--chromo',
            help='Chromosome')
        p.add_argument(
            '--start',
            help='Start position',
            type=int)
        p.add_argument(
            '--end',
            help='End position',
            type=int)
        p.add_argument(
            '--nb_sample',
            help='Maximum # training samples',
            type=int)
        p.add_argument(
            '--nb_dropout',
            help='Number of dropout samples',
            type=int)
        p.add_argument(
            '--max_mem',
            help='Maximum memory load',
            type=int,
            default=14000)
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

        log.info('Load data')
        def read_data(path):
            f = open_hdf(path, cache_size=opts.max_mem)
            data = dict()
            for k, v in f['data'].items():
                data[k] = v
            for k, v in f['pos'].items():
                data[k] = v
            return (f, data)

        labels = read_labels(opts.data_file)
        data_file, data = read_data(opts.data_file)

        if opts.chromo is not None:
            sel = data['chromo'].value == str(opts.chromo).encode()
            if opts.start is not None:
                sel &= data['pos'].value >= opts.start
            if opts.end is not None:
                sel &= data['pos'].value <= opts.end
            if sel.sum() == 0:
                log.warn('No samples satisfy filter!')
                return 0
            log.info('Selecting %d samples' % (sel.sum()))
            for k in data.keys():
                if len(data[k].shape) > 1:
                    data[k] = data[k][sel, :]
                else:
                    data[k] = data[k][sel]

        def to_view(d):
            for k in d.keys():
                d[k] = ArrayView(d[k], stop=opts.nb_sample)

        to_view(data)
        log.info('%d samples' % (list(data.values())[0].shape[0]))

        log.info('Load model')
        model = load_model(opts.model_file, opts.model_weights_file)

        def progress(batch, nb_batch):
            batch += 1
            c = max(1, int(np.ceil(nb_batch / 50)))
            if batch == 1 or batch == nb_batch or batch % c == 0:
                print('%5d / %d (%.1f%%)' % (batch, nb_batch,
                                             batch / nb_batch * 100))

        def predict(dropout=False):
            z = model.predict(data, verbose=opts.verbose,
                            callbacks=[progress], dropout=dropout)
            return z

        if opts.nb_dropout:
            log.info('Using dropout')
            model.compile(loss=model.loss, optimizer=model.optimizer,
                          dropout=True)
            for i in range(opts.nb_dropout):
                log.info('Predict (%d / %d)' % (i + 1, opts.nb_dropout))
                z = predict(dropout=True)
                write_z(data, z, labels, opts.out_file,
                        unlabeled=True, name='z%d' % (i))
        else:
            log.info('Predict')
            z = predict()
            write_z(data, z, labels, opts.out_file,
                    unlabeled=True, name='z')

        data_file.close()
        log.info('Done!')

        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
