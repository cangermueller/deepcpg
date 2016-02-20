#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import numpy as np

import predict.models.dnn.model as mod


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
            description='Convert model files')
        p.add_argument(
            'model',
            help='Model files',
            nargs='+')
        p.add_argument(
            '-j', '--out_json',
            help='Output json file')
        p.add_argument(
            '-w', '--out_weights',
            help='Output weights file')
        p.add_argument(
            '-p', '--out_pickle',
            help='Output pickle file')
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
        sys.setrecursionlimit(10**6)

        log.info('Load model')
        model = mod.model_from_list(opts.model)

        log.info('Save model')
        if opts.out_json is not None:
            mod.model_to_json(model, opts.out_json)
        if opts.out_weights is not None:
            model.save_weights(opts.out_weights, overwrite=True)
        if opts.out_pickle is not None:
            mod.model_to_pickle(model, opts.out_pickle)

        log.info('Done!')
        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
