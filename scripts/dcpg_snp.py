#!/usr/bin/env python

"""Compute the effect of DNA mutations on methylation.

Computes the effect of DNA mutation on the mean methylation rate or
cell-to-cell variance using gradient backpropagation.

Examples
--------
Compute the effect on mean methylation rates and cell-to-cell variance:

.. code:: bash

    dcpg_snp.py
        ./data/*.h5
        --model_files ./model/dna
        --out_file ./effects.h5
        --targets mean var

Compute weighted mean effects in DNA sequence windows of length 101:

.. code:: bash

    dcpg_snp.py
        ./data/*.h5
        --model_files ./model/dna
        --out_file ./effects.h5
        --targets mean var
        --dna_wlen 101
        --agg_effects wmean
"""

import sys
import os

import argparse
import h5py as h5
from keras import backend as K
import numpy as np
import logging

from deepcpg import data as dat
from deepcpg import models as mod
from deepcpg.data import hdf
from deepcpg.utils import ProgressBar, linear_weights


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
            description='Computes effects of DNA mutations by gradient ' +
            'backpropagation')
        p.add_argument(
            'data_files',
            help='Input data files',
            nargs='+')
        p.add_argument(
            '--model_files',
            help='Model files',
            nargs='+')
        p.add_argument(
            '--replicate_names',
            help='Regex to select replicates',
            nargs='+')
        p.add_argument(
            '--nb_replicate',
            type=int,
            help='Maximum number of replicates')
        p.add_argument(
            '-o', '--out_file',
            help='Output HDF5 file with mutation effects')
        p.add_argument(
            '--store_inputs',
            help='Store model inputs in output file',
            action='store_true')

        p.add_argument(
            '--targets',
            help='Targets on which the effect of DNA mutation is computed',
            choices=['mean', 'var'],
            default=['mean'],
            nargs='+')
        p.add_argument(
            '--agg_effects',
            help='Function to aggregate effects along in DNA sequence window',
            choices=['mean', 'wmean', 'max'])
        p.add_argument(
            '--dna_wlen',
            help='Maximum length of input DNA sequence window',
            type=int)

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
            help='Seed of random number generator',
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

        log.info('Loading model ...')
        K.set_learning_phase(0)
        model = mod.load_model(opts.model_files)

        # Get DNA layer.
        dna_layer = None
        for i, name in enumerate(model.input_names):
            if name == 'dna':
                dna_layer = model.input_layers[i]
                break
        if not dna_layer:
            raise ValueError('The provided model is not a DNA model!')

        # Create output vector.
        outputs = []
        for output in model.outputs:
            outputs.append(K.reshape(output, (-1, 1)))
        outputs = K.concatenate(outputs, axis=1)

        # Compute gradient of outputs wrt. DNA layer.
        grads = []
        for name in opts.targets:
            if name == 'mean':
                target = K.mean(outputs, axis=1)
            elif name == 'var':
                target = K.var(outputs, axis=1)
            else:
                raise ValueError('Invalid effect size "%s"!' % name)
            grad = K.gradients(target, dna_layer.output)
            grads.extend(grad)
        grad_fun = K.function(model.inputs, grads)

        log.info('Reading data ...')
        nb_sample = dat.get_nb_sample(opts.data_files, opts.nb_sample)
        replicate_names = dat.get_replicate_names(
            opts.data_files[0],
            regex=opts.replicate_names,
            nb_key=opts.nb_replicate)
        data_reader = mod.data_reader_from_model(
            model, outputs=False, replicate_names=replicate_names)
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

        log.info('Computing effects ...')
        progbar = ProgressBar(nb_sample, log.info)
        idx = 0
        for inputs in data_reader:
            if isinstance(inputs, dict):
                inputs = list(inputs.values())
            batch_size = len(inputs[0])
            progbar.update(batch_size)

            # Compute gradients.
            grads = grad_fun(inputs)

            # Slice window at center.
            if opts.dna_wlen:
                for i, grad in enumerate(grads):
                    delta = opts.dna_wlen // 2
                    ctr = grad.shape[1] // 2
                    grads[i] = grad[:, (ctr-delta):(ctr+delta+1)]

            # Aggregate effects in window
            if opts.agg_effects:
                for i, grad in enumerate(grads):
                    if opts.agg_effects == 'mean':
                        grad = grad.mean(axis=1)
                    elif opts.agg_effects == 'wmean':
                        weights = linear_weights(grad.shape[1])
                        grad = np.average(grad, axis=1, weights=weights)
                    elif opts.agg_effects == 'max':
                        grad = grad.max(axis=1)
                    else:
                        tmp = 'Invalid function "%s"!' % (opts.agg_effects)
                        raise ValueError(tmp)
                    grads[i] = grad

            # Write computed effects
            for name, grad in zip(opts.targets, grads):
                h5_dump(name, grad, idx)

            # Store inputs
            if opts.store_inputs:
                for name, value in zip(model.input_names, inputs):
                    h5_dump(name, value, idx)

            # Store positions
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
