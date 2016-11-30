#!/usr/bin/env python

import sys
import os

import argparse
import logging
from keras.models import Model

from deepcpg import models as mod


def remove_outputs(model):
    while model.layers[-1] in model.output_layers:
        model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []


def rename_layers(model):
    for layer in model.layers:
        if layer in model.input_layers or layer.name.startswith(model.name):
            continue
        layer.name = '%s/%s' % (model.name, layer.name)


class App(object):

    def run(self, args):
        name = os.path.basename(args[0])
        parser = self.create_parser(name)
        opts = parser.parse_args(args[1:])
        return self.main(name, opts)

    def create_parser(self, name):
        p = argparse.ArgumentParser(
            prog=name,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Build a model')
        p.add_argument(
            '--dna_model',
            help='DNA model files or name',
            nargs='+')
        p.add_argument(
            '--cpg_model',
            help='CpG model files or name',
            nargs='+')
        p.add_argument(
            '--joint_model',
            help='Name of joint model',
            default='joint01')
        p.add_argument(
            '-o', '--out_model',
            help='Output model file')
        p.add_argument(
            '--out_json',
            help='Output JSON file')
        p.add_argument(
            '--out_weights',
            help='Output weights file')
        p.add_argument(
            '--output_names',
            help='Output names of new model. Will be reused from input ' +
            'models if empty.')
        p.add_argument(
            '--dropout',
            help='Dropout rate',
            type=float,
            default=0.0)
        p.add_argument(
            '--l1_decay',
            help='L1 weight decay',
            type=float,
            default=0.0)
        p.add_argument(
            '--l2_decay',
            help='L2 weight decay',
            type=float,
            default=0.0)
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

        if not opts.dna_model and not opts.cpg_model:
            raise ValueError('No input models given!')

        dna_model = None
        cpg_model = None
        output_names = opts.output_names

        if opts.dna_model:
            log.info('Loading DNA model ...')
            dna_model = mod.load_model(opts.dna_model)
            if not output_names:
                output_names = dna_model.output_names
            remove_outputs(dna_model)

        if opts.cpg_model:
            log.info('Loading CpG model ...')
            cpg_model = mod.load_model(opts.cpg_model)
            if not output_names:
                output_names = cpg_model.output_names
            remove_outputs(cpg_model)

        if dna_model is not None and cpg_model is not None:
            log.info('Building joint model ...')
            rename_layers(dna_model)
            rename_layers(cpg_model)
            joint_model_builder = mod.joint.get(opts.joint_model)(
                l1_decay=opts.l1_decay,
                l2_decay=opts.l2_decay,
                dropout=opts.dropout)
            stem = joint_model_builder([dna_model, cpg_model])
            stem.name = '_'.join([stem.name, dna_model.name, cpg_model.name])
        elif dna_model is not None:
            stem = dna_model
        else:
            stem = cpg_model

        log.info('Adding outputs ...')
        outputs = mod.add_output_layers(stem.outputs, output_names)
        model = Model(input=stem.inputs, output=outputs, name=stem.name)
        model.summary()

        log.info('Storing model ...')
        if opts.out_model:
            model.save(opts.out_model)
        if opts.out_json:
            mod.save_model(model, opts.out_json)
        if opts.out_weights:
            model.save_weights(opts.out_weights)

        log.info('Done!')
        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
