#!/usr/bin/env python

"""Compute filter activations of a DeepCpG model.

Computes the activation of the filters of the first convolutional layer for a
given DNA model. The resulting activations can be used to visualize and cluster
motifs, or correlated with model outputs.

Examples
--------
Compute activations in 25000 sequence windows and also store DNA sequences.
For example to visualize motifs.

.. code:: bash

    dcpg_filter_act.py
        ./data/*.h5
        --model_files ./models/dna
        --out_file ./activations.h5
        --nb_sample 25000
        --store_inputs

Compute the weighted mean activation in each sequence window and also store
model predictions. For example to cluster motifs or to correlated mean motif
activations with model predictions.

.. code:: bash

    dcpg_filter_act.py
        ./data/*.h5
        --model_files ./models/dna
        --out_file ./activations.h5
        --act_fun wmean

See Also
--------
* ``dcpg_filter_motifs.py``: For motif visualization and analysis.
"""

from __future__ import print_function
from __future__ import division

import sys
import os

import argparse
import h5py as h5
from keras import backend as K
import numpy as np
import logging
import six

from deepcpg import data as dat
from deepcpg import models as mod
from deepcpg.data import hdf, dna
from deepcpg.utils import ProgressBar, to_list, linear_weights


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
            description='Computes filter activations of a DeepCpG model')
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
            help='Output file',
            default='activations.h5')

        g = p.add_argument_group('arguments for summarizing activations')
        g.add_argument(
            '--act_fun',
            help='Function for summarizing activations in each sequence window',
            choices=['mean', 'wmean', 'max'])
        g.add_argument(
            '--act_wlen',
            help='Maximal length of sequence windows',
            type=int)

        g = p.add_argument_group('output arguments')
        g.add_argument(
            '--store_outputs',
            help='Store output labels',
            action='store_true')
        g.add_argument(
            '--store_preds',
            help='Store model predictions',
            action='store_true')
        g.add_argument(
            '--store_inputs',
            help='Store model inputs',
            action='store_true')

        g = p.add_argument_group('advanced arguments')
        g.add_argument(
            '--nb_sample',
            help='Number of samples',
            type=int)
        g.add_argument(
            '--shuffle',
            help='Randomly sample inputs',
            action='store_true')
        g.add_argument(
            '--batch_size',
            help='Batch size',
            type=int,
            default=128)
        g.add_argument(
            '--seed',
            help='Seed of random number generator',
            type=int,
            default=0)
        g.add_argument(
            '--verbose',
            help='More detailed log messages',
            action='store_true')
        g.add_argument(
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

        log.info('Loading model ...')
        K.set_learning_phase(0)
        model = mod.load_model(opts.model_files, log=log.info)

        weight_layer, act_layer = mod.get_first_conv_layer(model.layers, True)
        log.info('Using activation layer "%s"' % act_layer.name)
        log.info('Using weight layer "%s"' % weight_layer.name)

        try:
            dna_idx = model.input_names.index('dna')
        except BaseException:
            raise IOError('Model is not a valid DNA model!')

        fun_outputs = to_list(act_layer.output)
        if opts.store_preds:
            fun_outputs += to_list(model.output)
        fun = K.function([to_list(model.input)[dna_idx]], fun_outputs)

        log.info('Reading data ...')
        if opts.store_outputs or opts.store_preds:
            output_names = model.output_names
        else:
            output_names = None
        data_reader = mod.DataReader(
            output_names=output_names,
            use_dna=True,
            dna_wlen=to_list(model.input_shape)[dna_idx][1]
        )
        nb_sample = dat.get_nb_sample(opts.data_files, opts.nb_sample)
        data_reader = data_reader(opts.data_files,
                                  nb_sample=nb_sample,
                                  batch_size=opts.batch_size,
                                  loop=False,
                                  shuffle=opts.shuffle)

        meta_reader = hdf.reader(opts.data_files, ['chromo', 'pos'],
                                 nb_sample=nb_sample,
                                 batch_size=opts.batch_size,
                                 loop=False,
                                 shuffle=False)

        out_file = h5.File(opts.out_file, 'w')
        out_group = out_file

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

        log.info('Computing activations')
        progbar = ProgressBar(nb_sample, log.info)
        idx = 0
        for data in data_reader:
            if isinstance(data, tuple):
                inputs, outputs, weights = data
            else:
                inputs = data
            if isinstance(inputs, dict):
                inputs = list(inputs.values())
            batch_size = len(inputs[0])
            progbar.update(batch_size)

            if opts.store_inputs:
                for i, name in enumerate(model.input_names):
                    h5_dump('inputs/%s' % name,
                            dna.onehot_to_int(inputs[i]), idx)

            if opts.store_outputs:
                for name, output in six.iteritems(outputs):
                    h5_dump('outputs/%s' % name, output, idx)

            fun_eval = fun(inputs)
            act = fun_eval[0]

            if opts.act_wlen:
                delta = opts.act_wlen // 2
                ctr = act.shape[1] // 2
                act = act[:, (ctr-delta):(ctr+delta+1)]

            if opts.act_fun:
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

            for name, value in six.iteritems(next(meta_reader)):
                h5_dump(name, value, idx)

            idx += batch_size
        progbar.close()

        out_file.close()
        log.info('Done!')

        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
