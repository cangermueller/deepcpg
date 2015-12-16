#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import numpy as np
import pickle

from predict.models.dnn.utils import load_model

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
            description='Inspect model training')
        p.add_argument(
            'model_json',
            help='Model json file')
        p.add_argument(
            'model_weights',
            help='Model weights file')
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

        log.info('Load model')
        model = load_model(opts.model_json, opts.model_weights)

        sys.setrecursionlimit(10**6)
        log.info('Pickle model')
        with open(pt.join(opts.out_file), 'wb') as f:
            pickle.dump(model, f)

        log.info('Done!')
        return 0


if __name__ == '__main__':
    app=App()
    app.run(sys.argv)
