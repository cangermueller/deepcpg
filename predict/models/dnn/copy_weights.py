#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import numpy as np
import random
import re

import predict.models.dnn.model as mod


def copy_weights(src_nodes, dst_nodes, nodes=None):
    for src_name, src_node in src_nodes.items():
        if nodes is not None and src_name not in nodes:
            continue
        dst_name = src_name
        if re.match('^u\d+_o', dst_name):
            dst_name = dst_name.replace('u', 'c')
        if dst_name not in dst_nodes:
            continue
        src_node.set_weights(dst_nodes[dst_name])


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
            'dst_model',
            help='Destination model',
            nargs='+')
        p.add_argument(
            '--src_model',
            help='Source model')
        p.add_argument(
            '--src_nodes',
            help='Only copy weights from these nodes')
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

        log.info('Loading destination model')
        dst_model = mod.model_from_list(opts.dst_model, compile=False)

        log.info('Loading source model')
        src_model = mod.model_from_list(opts.src_model, compile=False)

        log.info('Copy weights')
        copy_weights(src_model, dst_model, opts.nodes)

        log.info('Save model')
        if opts.out_json is not None:
            mod.model_to_json(dst_model, pt.join(opts.out_dir, 'model.json'))
        if opts.out_weights is not None:
            dst_model.save_weights(opts.out_weights)
        if opts.out_pickle is not None:
            mod.model_to_pickle(dst_model, opts.out_pickle)

        log.info('Done!')
        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
