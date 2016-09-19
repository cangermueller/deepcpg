#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import pandas as pd
import numpy as np

import predict.models.dnn.utils as ut
import predict.models.dnn.model as mod


def select_data(data, chromo, start=None, end=None):
    sel = data['chromo'].value == str(chromo).encode()
    if start is not None:
        sel &= data['pos'].value >= start
    if end is not None:
        sel &= data['pos'].value <= end
    for k in data.keys():
        if len(data[k].shape) > 1:
            data[k] = data[k][sel, :]
        else:
            data[k] = data[k][sel]
    return sel


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
            '--batch_size',
            help='Batch size',
            type=int,
            default=128)
        p.add_argument(
            '--nb_sample',
            help='Maximum # training samples',
            type=int)
        p.add_argument(
            '--all',
            help='Include unlabeled sites',
            action='store_true')
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

        log.info('Load model')
        model = mod.model_from_list(opts.model)

        log.info('Load data')
        targets = ut.read_targets(opts.data_file)
        data_file, data = ut.read_hdf(opts.data_file, opts.max_mem)
        if opts.chromo is not None:
            log.info('Select data')
            select_data(data, opts.chromo, opts.start, opts.end)
            log.info('%d sites selected' % (len(data['pos'])))
        ut.to_view(data, stop=opts.nb_sample)

        print('%d samples' % (list(data.values())[0].shape[0]))
        print()

        def progress(*args, **kwargs):
            h = mod.progress(*args, **kwargs)
            if h is not None:
                print(h)

        log.info('Predict')
        z = model.predict(data, verbose=opts.verbose,
                          callbacks=[progress],
                          batch_size=opts.batch_size)
        log.info('Write')
        ut.write_z(data, z, targets, opts.out_file, unlabeled=opts.all)

        data_file.close()
        log.info('Done!')

        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
