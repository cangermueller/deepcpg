#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import numpy as np
import random
import re

import predict.models.dnn.model as mod
import predict.utils as pu


def copy_weights(src_nodes, dst_nodes, nodes=None):
    nb_copy = 0
    for src_name, src_node in src_nodes.items():
        if nodes is not None:
            if len(pu.filter_regex(src_name, nodes)) == 0:
                continue
        dst_name = src_name
        if re.match('^u\d+_o', dst_name):
            dst_name = dst_name.replace('u', 'c')
        if dst_name not in dst_nodes:
            continue
        print('%s -> %s' % (src_name, dst_name))
        dst_nodes[dst_name].set_weights(src_node.get_weights())
        nb_copy += 1
    return nb_copy


def check_weights(src_nodes, dst_nodes, nodes=None):
    if nodes is None:
        nodes = dst_nodes.keys()
    for n in nodes:
        ws = src_nodes[n].get_weights()
        wd = dst_nodes[n].get_weights()
        if not isinstance(ws, list):
            ws = list(ws)
            wd = list(wd)
        for i in range(len(ws)):
            assert np.all(ws[i] == wd[i])


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
            '-s', '--src_model',
            help='Source model',
            nargs='+')
        p.add_argument(
            '-d', '--dst_model',
            help='Destination model',
            nargs='+')
        p.add_argument(
            '-n', '--nodes',
            help='Only copy weights from these nodes',
            nargs='+')
        p.add_argument(
            '-c', '--compile',
            help='Compile model',
            action='store_true')
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
            random.seed(opts.seed)
        sys.setrecursionlimit(10**6)

        if opts.src_model is None:
            raise 'No source model given!'
        if opts.dst_model is None:
            raise 'No destination model given!'

        log.info('Loading source model')
        src_model = mod.model_from_list(opts.src_model, compile=False)

        log.info('Loading destination model')
        dst_model = mod.model_from_list(opts.dst_model, compile=False)

        log.info('Copy weights')
        h = copy_weights(src_model.nodes, dst_model.nodes, opts.nodes)
        log.info('Weights copied from %d nodes' % (h))
        #  check_weights(src_model.nodes, dst_model.nodes, opts.nodes)

        log.info('Save model')
        if opts.out_json is not None:
            mod.model_to_json(dst_model, opts.out_json)
        if opts.out_weights is not None:
            dst_model.save_weights(opts.out_weights, overwrite=True)
        if opts.out_pickle is not None:
            mod.model_to_pickle(dst_model, opts.out_pickle)

        log.info('Done!')
        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
