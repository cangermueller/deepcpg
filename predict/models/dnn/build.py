#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import numpy as np
import h5py as h5
import random

import predict.models.dnn.model as mod
import predict.models.dnn.utils as ut
from predict.models.dnn.params import Params


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
            description='Build model')
        p.add_argument(
            'param_file',
            help='Model parameter file')
        p.add_argument(
            'data_file',
            help='Data file')
        p.add_argument(
            '-o', '--out_dir',
            help='Output directory')
        p.add_argument(
            '--targets',
            help='Target names',
            nargs='+')
        p.add_argument(
            '--compile',
            help='Compile model',
            action='store_true')
        p.add_argument(
            '--out_json',
            help='Output json file',
            nargs='?',
            default='./model.json')
        p.add_argument(
            '--out_weights',
            help='Output weights file',
            nargs='?',
            default='./model_weights.h5')
        p.add_argument(
            '--out_pickle',
            help='Output pickle file',
            nargs='?',
            default='./model.h5')
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
            random.seed(opts.seed)
        sys.setrecursionlimit(10**6)

        log.info('Initialize')
        targets = ut.read_targets(opts.data_file, opts.targets)
        model_params = Params.from_yaml(opts.param_file)
        seq_len = None
        cpg_len = None
        nb_unit = None
        f = h5.File(opts.data_file)
        g = f['data']
        if 's_x' in g:
            seq_len = g['s_x'].shape[1]
        if 'c_x' in g:
            nb_unit = g['c_x'].shape[2]
            cpg_len = g['c_x'].shape[3]
        f.close()

        log.info('Build mode')
        model = mod.build(model_params, targets['id'], seq_len, cpg_len,
                          nb_unit=nb_unit, compile=opts.compile)

        log.info('Save model')
        if opts.out_json is not None:
            mod.model_to_json(model, pt.join(opts.out_dir, 'model.json'))
        if opts.out_weights is not None:
            model.save_weights(opts.out_weights)
        if opts.out_pickle is not None:
            mod.model_to_pickle(model, opts.out_pickle)

        log.info('Done!')
        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
