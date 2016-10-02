#!/usr/bin/env python

import sys
import os

import argparse
import pandas as pd
import logging

from deepcpg import data as dat
from deepcpg import evaluation as ev
from deepcpg import models as mod
from deepcpg.utils import ProgressBar


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
            description='Tests the performance of a model')
        p.add_argument(
            'data_files',
            nargs='+',
            help='Test data files')
        p.add_argument(
            '--model_files',
            nargs='+',
            help='Model files')
        p.add_argument(
            '-o', '--out_summary',
            help='Output summary file')
        p.add_argument(
            '--out_data',
            help='Output file with predictions and labels')
        p.add_argument(
            '--replicate_names',
            help='List of regex to filter CpG context units',
            nargs='+')
        p.add_argument(
            '--batch_size',
            help='Batch size',
            type=int,
            default=128)
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
        p.add_argument(
            '--data_nb_worker',
            help='Number of worker for data generator queue',
            type=int,
            default=1)
        return p

    def main(self, name, opts):
        logging.basicConfig(filename=opts.log_file,
                            format='%(levelname)s (%(asctime)s): %(message)s')
        log = logging.getLogger(name)
        if opts.verbose:
            log.setLevel(logging.DEBUG)
        else:
            log.setLevel(logging.INFO)

        log.info('Loading model ...')
        model = mod.load_model(opts.model_files)
        model_builder = mod.get_class(model.name)

        if model.name.lower().startswith('dna'):
            dna_wlen = int(model.inputs[0].get_shape()[1])
            model_builder = model_builder(dna_wlen=dna_wlen)

        elif model.name.lower().startswith('cpg'):
            cpg_wlen = int(model.inputs[0].get_shape()[2])
            replicate_names = dat.h5_ls(opts.data_files[0], 'inputs/cpg',
                                        opts.replicate_names)
            model_builder = model_builder(replicate_names,
                                          cpg_wlen=cpg_wlen)

        else:
            dna_wlen = int(model.inputs[0].get_shape()[1])
            cpg_wlen = int(model.inputs[0].get_shape()[2])
            replicate_names = dat.h5_ls(opts.data_files[0], 'inputs/cpg',
                                        opts.replicate_names)
            model_builder = model_builder(replicate_names,
                                          cpg_wlen=cpg_wlen,
                                          dna_wlen=dna_wlen)

        log.info('Reading data ...')
        nb_sample = dat.get_nb_sample(opts.data_files, opts.nb_sample)

        meta_reader = dat.h5_reader(opts.data_files, ['chromo', 'pos'],
                                    nb_sample=nb_sample,
                                    batch_size=opts.batch_size,
                                    loop=False, shuffle=False)

        data_reader = model_builder.reader(opts.data_files,
                                           output_names=model.output_names,
                                           nb_sample=nb_sample,
                                           batch_size=opts.batch_size,
                                           loop=False, shuffle=False)

        log.info('Predicting ...')
        data = dict()
        progbar = ProgressBar(nb_sample, log.info)
        for inputs, outputs, weights in data_reader:
            batch_size = len(list(inputs.values())[0])
            progbar.update(batch_size)

            preds = model.predict(inputs)
            pred = [pred.squeeze() for pred in preds]

            data_batch = dict()
            data_batch['preds'] = dict()
            for name, pred in zip(model.output_names, preds):
                data_batch['preds'][name] = pred
            data_batch['outputs'] = outputs
            data_batch['weights'] = weights
            for name, value in next(meta_reader).items():
                data_batch[name] = value
            dat.add_to_dict(data_batch, data)

        progbar.close()

        data = dat.stack_dict(data)

        perf = []
        for output in model.output_names:
            tmp = ev.evaluate(data['outputs'][output], data['preds'][output])
            perf.append(pd.DataFrame(tmp, index=[output]))
        perf = pd.concat(perf)
        mean = perf.mean()
        mean.name = 'mean'
        perf.append(mean)
        perf.index.name = 'output'
        perf.reset_index(inplace=True)

        print(perf.to_string())
        if opts.out_summary:
            perf.to_csv(opts.out_summary, sep='\t', index=False)

        if opts.out_data:
            dat.h5_write_data(data, opts.out_data)

        log.info('Done!')

        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
