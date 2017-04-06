#!/usr/bin/env python

"""Evaluate the prediction performance of a DeepCpG model.

Imputes missing methylation states and evaluates model on observed states.
``--out_report`` will write evaluation metrics to a TSV file using.
``--out_data`` will write predicted and observed methylation state to a HDF5
file with following structure:

* ``chromo``: The chromosome of the CpG site.
* ``pos``: The position of the CpG site on the chromosome.
* ``outputs``: The input methylation state of each cell and CpG site, which \
    can either observed or missing (-1).
* ``preds``: The predicted methylation state of each cell and CpG site.

Examples
--------

.. code:: bash

    dcpg_eval.py
        ./data/*.h5
        --model_files ./model
        --out_data ./eval/data.h5
        --out_report ./eval/report.tsv
"""

from __future__ import print_function
from __future__ import division

import os
import random
import sys

import argparse
import h5py as h5
import logging
import numpy as np
import pandas as pd
import six

from deepcpg import data as dat
from deepcpg import evaluation as ev
from deepcpg import models as mod
from deepcpg.data import hdf
from deepcpg.utils import ProgressBar, to_list


class H5Writer(object):

    def __init__(self, filename, nb_sample):
        self.out_file = h5.File(filename, 'w')
        self.nb_sample = nb_sample
        self.idx = 0

    def __call__(self, name, data, dtype=None, compression='gzip', stay=False):
        if name not in self.out_file:
            if dtype is None:
                dtype = data.dtype
            self.out_file.create_dataset(
                name=name,
                shape=[self.nb_sample] + list(data.shape[1:]),
                dtype=dtype,
                compression=compression
            )
        self.out_file[name][self.idx:(self.idx + len(data))] = data
        if not stay:
            self.idx += len(data)

    def write_dict(self, data, name='', level=0, *args, **kwargs):
        size = None
        for key, value in six.iteritems(data):
            _name = '%s/%s' % (name, key)
            if isinstance(value, dict):
                self.write_dict(value, name=_name, level=level + 1,
                                *args, **kwargs)
            else:
                if size:
                    assert size == len(value)
                else:
                    size = len(value)
                self(_name, value, stay=True, *args, **kwargs)
        if level == 0:
            self.idx += size

    def close(self):
        self.out_file.close()


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
            description='Evaluates prediction performance of a DeepCpG model')
        p.add_argument(
            'data_files',
            help='Input data files for evaluation',
            nargs='+')
        p.add_argument(
            '--model_files',
            help='Model files',
            nargs='+')
        p.add_argument(
            '-o', '--out_report',
            help='Output report file with evaluation metrics')
        p.add_argument(
            '--out_data',
            help='Output file with predictions and labels')
        p.add_argument(
            '--replicate_names',
            help='Regex to select replicates',
            nargs='+')
        p.add_argument(
            '--nb_replicate',
            type=int,
            help='Maximum number of replicates')
        p.add_argument(
            '--eval_size',
            help='Maximum number of samples that are kept in memory for'
            ' batch-wise evaluation. If zero, evaluate on entire data set.',
            type=int,
            default=100000)
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
            '--nb_sample',
            help='Number of samples',
            type=int)
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

        if not opts.model_files:
            raise ValueError('No model files provided!')

        log.info('Loading model ...')
        model = mod.load_model(opts.model_files)

        log.info('Loading data ...')
        nb_sample = dat.get_nb_sample(opts.data_files, opts.nb_sample)
        replicate_names = dat.get_replicate_names(
            opts.data_files[0],
            regex=opts.replicate_names,
            nb_key=opts.nb_replicate)
        data_reader = mod.data_reader_from_model(
            model, replicate_names, replicate_names=replicate_names)

        # Seed used since unobserved input CpG states are randomly sampled
        if opts.seed is not None:
            np.random.seed(opts.seed)
            random.seed(opts.seed)

        data_reader = data_reader(opts.data_files,
                                  nb_sample=nb_sample,
                                  batch_size=opts.batch_size,
                                  loop=False, shuffle=False)

        meta_reader = hdf.reader(opts.data_files, ['chromo', 'pos'],
                                 nb_sample=nb_sample,
                                 batch_size=opts.batch_size,
                                 loop=False, shuffle=False)

        writer = None
        if opts.out_data:
            writer = H5Writer(opts.out_data, nb_sample)

        log.info('Predicting ...')
        nb_tot = 0
        nb_eval = 0
        data_eval = dict()
        perf_eval = []
        progbar = ProgressBar(nb_sample, log.info)
        for inputs, outputs, weights in data_reader:
            batch_size = len(list(inputs.values())[0])
            nb_tot += batch_size
            progbar.update(batch_size)

            preds = to_list(model.predict(inputs))

            data_batch = dict()
            data_batch['preds'] = dict()
            data_batch['outputs'] = dict()
            for i, name in enumerate(model.output_names):
                data_batch['preds'][name] = preds[i].squeeze()
                data_batch['outputs'][name] = outputs[name].squeeze()

            for name, value in six.iteritems(next(meta_reader)):
                data_batch[name] = value

            if writer:
                writer.write_dict(data_batch)

            nb_eval += batch_size
            dat.add_to_dict(data_batch, data_eval)

            if nb_tot >= nb_sample or \
                    (opts.eval_size and nb_eval >= opts.eval_size):
                data_eval = dat.stack_dict(data_eval)
                perf_eval.append(ev.evaluate_outputs(data_eval['outputs'],
                                                     data_eval['preds']))
                data_eval = dict()
                nb_eval = 0

        progbar.close()
        if writer:
            writer.close()

        report = pd.concat(perf_eval)
        report = report.groupby(['metric', 'output']).mean().reset_index()

        if opts.out_report:
            report.to_csv(opts.out_report, sep='\t', index=False)

        report = ev.unstack_report(report)
        print(report.to_string())

        log.info('Done!')

        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
