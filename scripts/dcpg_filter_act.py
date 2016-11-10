#!/usr/bin/env python

import sys
import os

import argparse
import h5py as h5
from keras import backend as K
from keras import layers as kl
import numpy as np
import logging

from deepcpg import data as dat
from deepcpg import models as mod
from deepcpg.data import hdf, dna
from deepcpg.utils import ProgressBar, to_list


def linear_weights(wlen, start=0.1):
    w = np.linspace(start, 1, np.ceil(wlen / 2))
    v = w
    if wlen % 2:
        v = v[:-1]
    w = np.hstack((w, v[::-1]))
    return (w)


def get_layer_by_depth(layers, depth=1, layer_class=kl.Activation):
    idx = 1
    for layer in layers:
        if isinstance(layer, layer_class):
            if idx == depth:
                return layer
            else:
                idx += 1
    return None


class App(object):

    def run(self, args):
        name = os.path.basename(args[0])
        parser = self.create_parser(name)
        opts = parser.parse_args(args[1:])
        self.opts = opts
        return self.main(name, opts)

    def create_parser(self, name):
        p = argparse.ArgumentParser(
            prog=name,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Compute filter activations')
        p.add_argument(
            'data_files',
            help='Data files',
            nargs='+')
        p.add_argument(
            '--model_files',
            help='Model files',
            nargs='+')
        p.add_argument(
            '-o', '--out_file',
            help='Output file')
        p.add_argument(
            '--act_layer',
            help='Name or depth of activation layer',
            default='1')
        p.add_argument(
            '--weight_layer',
            help='Name or depth of weight layer',
            default='1')
        p.add_argument(
            '--act_fun',
            help='Function applied to reduce sequence window activations',
            choices=['mean', 'wmean', 'max'])
        p.add_argument(
            '--act_wlen',
            help='Slice wlen at center',
            type=int)
        p.add_argument(
            '--store_preds',
            help='Store model predictions',
            action='store_true')
        p.add_argument(
            '--store_inputs',
            help='Store model inputs',
            action='store_true')
        p.add_argument(
            '--nb_sample',
            help='Number of samples',
            type=int)
        p.add_argument(
            '--batch_size',
            help='Batch size',
            type=int,
            default=128)
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

        if not opts.model_files:
            raise ValueError('No model files provided!')

        if opts.act_wlen is not None and opts.act_wlen % 2 == 0:
            raise ValueError('Activation window length must be divisible by ' +
                             'two!')

        log.info('Loading model ...')
        K.set_learning_phase(0)
        model = mod.load_model(opts.model_files)

        if opts.act_layer.isdigit():
            act_layer = get_layer_by_depth(model.layers, int(opts.act_layer),
                                           layer_class=kl.Activation)
        else:
            act_layer = model.get_layer(opts.act_layer)
        if not act_layer:
            raise ValueError('Activation layer not found!')
        log.info('Using activation layer "%s"' % act_layer.name)
        fun_outputs = to_list(act_layer.output)
        if opts.store_preds:
            fun_outputs += to_list(model.output)
        fun = K.function(to_list(model.input), fun_outputs)

        log.info('Reading data ...')
        nb_sample = dat.get_nb_sample(opts.data_files, opts.nb_sample)
        data_reader = mod.data_reader_from_model(model)
        data_reader = data_reader(opts.data_files,
                                  nb_sample=nb_sample,
                                  batch_size=opts.batch_size,
                                  loop=False,
                                  shuffle=False)

        meta_reader = hdf.reader(opts.data_files, ['chromo', 'pos'],
                                 nb_sample=nb_sample,
                                 batch_size=opts.batch_size,
                                 loop=False,
                                 shuffle=False)

        out_file = h5.File(opts.out_file, 'w')
        out_group = out_file

        if opts.weight_layer:
            if opts.weight_layer.isdigit():
                weight_layer = get_layer_by_depth(model.layers,
                                                  int(opts.weight_layer),
                                                  layer_class=kl.Conv1D)
            else:
                weight_layer = model.get_layer(opts.weight_layer)
            if not weight_layer:
                raise ValueError('Weight layer not found!')
            log.info('Using weight layer "%s"' % weight_layer.name)
            weights = weight_layer.get_weights()
            out_group['weights/weights'] = weights[0]
            out_group['weights/bias'] = weights[1]

        def h5_dump(path, data, idx, dtype=None, compression='gzip'):
            if path not in out_group:
                if dtype is None:
                    dtype = data.dtype
                out_group.create_dataset(
                    name=path,
                    shape=[nb_sample] + list(data.shape[1:]),
                    dtype=dtype,
                    compression=compression
                )
            out_group[path][idx:idx+len(data)] = data

        progbar = ProgressBar(nb_sample, log.info)
        idx = 0
        for inputs, outputs, weights in data_reader:
            if isinstance(inputs, dict):
                inputs = list(inputs.values())
            batch_size = len(inputs[0])
            progbar.update(batch_size)

            if opts.store_inputs:
                for i, name in enumerate(model.input_names):
                    h5_dump('inputs/%s' % name,
                            dna.onehot_to_int(inputs[i]), idx)

            fun_eval = fun(inputs)
            act = fun_eval[0]

            if opts.act_wlen is not None:
                delta = opts.act_wlen // 2
                ctr = act.shape[1] // 2
                act = act[:, (ctr-delta):(ctr+delta+1)]

            if opts.act_fun is not None:
                if opts.act_fun == 'mean':
                    act = act.mean(axis=1)
                elif opts.act_fun == 'wmean':
                    weights = linear_weights(act.shape[1])
                    act = np.average(act, axis=1, weights=weights)
                elif opts.act_fun == 'max':
                    act = act.max(axis=1)
                else:
                    raise ValueError('Invalid function "%s"!' % (opts.act_fun))

            h5_dump('act', act, idx)

            if opts.store_preds:
                preds = fun_eval[1:]
                for i, name in enumerate(model.output_names):
                    h5_dump('preds/%s' % name, preds[i].squeeze(), idx)

            for name, value in next(meta_reader).items():
                h5_dump(name, value, idx)

            idx += batch_size
        progbar.close()

        out_file.close()
        log.info('Done!')

        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
