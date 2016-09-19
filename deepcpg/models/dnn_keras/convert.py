#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt

import cpgkeras.layers.convolutional as kconv
import numpy as np

import predict.models.dnn.model as dmod
from predict.models.dnn.params import Params
import predict.models.dnn_keras.model as d2mod


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
            description='Converts a legacy model')
        p.add_argument(
            'graph_model',
            nargs=3,
            help='yaml, json, and HDF5 model file')
        p.add_argument(
            '-o', '--out_model',
            help='Output HDF5 model file')
        p.add_argument(
            '-j', '--out_json',
            help='Output json file')
        p.add_argument(
            '-w', '--out_weights',
            help='Output weights file')
        p.add_argument(
            '-c', '--compile',
            help='Compile model',
            action='store_true')
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

        model_params = Params.from_yaml(opts.graph_model[0])
        graph_model = dmod.model_from_list(opts.graph_model[1:], compile=False)

        def get_input_len(name):
            cfgs = graph_model.get_config()['input_config']
            length = None
            for cfg in cfgs:
                if cfg['name'] == name:
                    length = cfg['input_shape'][0]
                    break
            assert length
            return length

        seq_len = None
        cpg_len = None
        if 's_x' in graph_model.input_order:
            seq_len = get_input_len('s_x')
        if 'c_x' in graph_model.input_order:
            cpg_len = get_input_len('c_x')
        targets = [x.replace('_y', '') for x in graph_model.output_order]

        model = d2mod.build(model_params, targets,
                            seq_len=seq_len, cpg_len=cpg_len, compile=False)

        for src_name, src_node in graph_model.nodes.items():
            print(src_name)
            dst_layer = model.get_layer(src_name)
            assert dst_layer
            src_weights = src_node.get_weights()
            if isinstance(src_node, kconv.Convolution1D):
                conv_weights = src_weights[0]
                conv_weights = np.rollaxis(conv_weights, 2, 0)
                conv_weights = np.rollaxis(conv_weights, 3, 1)
                conv_weights = np.rollaxis(conv_weights, 3, 2)
                src_weights[0] = conv_weights
            dst_layer.set_weights(src_weights)

        log.info('Save model')
        model.save(opts.out_model)
        if opts.out_json is not None:
            d2mod.model_to_json(model, opts.out_json)
        if opts.out_weights is not None:
            model.save_weights(opts.out_weights, overwrite=True)

        log.info('Done!')

        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
