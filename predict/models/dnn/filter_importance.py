#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import numpy as np

from predict.models.dnn.utils import read_labels, open_hdf, ArrayView
import predict.models.dnn.model as mod
from utils import write_z


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
            '--model',
            help='Model files',
            nargs='+')
        p.add_argument(
            '-o', '--out_file',
            help='Output file')
        p.add_argument(
            '--conv_layer',
            help='Convolutional layer',
            default='s_c1')
        p.add_argument(
            '--filters',
            help='Filters to be tested',
            type=int,
            nargs='+')
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
            log.info('Select %d samples' % (sel.sum()))
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
        model = mod.model_from_list(opts.model, compile=True)

        def progress(batch, nb_batch):
            batch += 1
            c = max(1, int(np.ceil(nb_batch / 50)))
            if batch == 1 or batch == nb_batch or batch % c == 0:
                print('%5d / %d (%.1f%%)' % (batch, nb_batch,
                                             batch / nb_batch * 100))

        def predict(dropout=False, d=data):
            z = model.predict(d, verbose=opts.verbose,
                            callbacks=[progress], dropout=dropout)
            return z

        log.info('Predict')
        z = predict()
        write_z(data, z, labels, opts.out_file, unlabeled=True, name='z')

        conv = model.nodes[opts.conv_layer]
        filters, bias = conv.get_weights()
        filters_list = opts.filters
        if filters_list is None:
            filters_list = range(filters.shape[0])
        for i in filters_list:
            log.info('Kill filter %d' % (i))
            filters_x = filters.copy()
            filters_x[i] = filters_x[i].mean()
            conv.set_weights((filters_x, bias))
            z = predict()
            write_z(data, z, labels, opts.out_file, unlabeled=True,
                  name='z_f%03d' % (i))

        data_file.close()
        log.info('Done!')

        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
