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
            description='Evaluates the performance of a model')
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
            '--nb_replicate',
            type=int,
            help='Maximum number of replicates')
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

        if not opts.model_files:
            raise ValueError('No model files provided!')

        log.info('Loading model ...')
        model = mod.load_model(opts.model_files)

        log.info('Reading data ...')
        nb_sample = dat.get_nb_sample(opts.data_files, opts.nb_sample)

        meta_reader = dat.h5_reader(opts.data_files, ['chromo', 'pos'],
                                    nb_sample=nb_sample,
                                    batch_size=opts.batch_size,
                                    loop=False, shuffle=False)

        data_reader = mod.data_reader_from_model(model)
        data_reader = data_reader(opts.data_files,
                                  nb_sample=nb_sample,
                                  batch_size=opts.batch_size,
                                  loop=False,
                                  shuffle=False)

        log.info('Predicting ...')
        data = dict()
        progbar = ProgressBar(nb_sample, log.info)
        for inputs, outputs, weights in data_reader:
            batch_size = len(list(inputs.values())[0])
            progbar.update(batch_size)

            preds = model.predict(inputs)
            if not isinstance(preds, list):
                preds = [preds]
            pred = [pred.squeeze() for pred in preds]

            data_batch = dict()
            data_batch['preds'] = dict()
            for name, pred in zip(model.output_names, preds):
                data_batch['preds'][name] = pred
            data_batch['outputs'] = outputs
            for name, value in next(meta_reader).items():
                data_batch[name] = value
            dat.add_to_dict(data_batch, data)

        progbar.close()

        data = dat.stack_dict(data)

        perf = []
        for output in model.output_names:
            metrics = mod.get_eval_metrics(output)
            tmp = ev.evaluate(data['outputs'][output], data['preds'][output],
                              metrics=metrics)
            tmp = pd.DataFrame({'output': output,
                                'metric': list(tmp.keys()),
                                'value': list(tmp.values())})
            perf.append(tmp)
        perf = pd.concat(perf)
        perf = perf[['metric', 'output', 'value']]
        perf.sort_values(['metric', 'value'], inplace=True)

        _perf = pd.pivot_table(perf, index='output', columns='metric',
                               values='value')
        _perf.reset_index('output', inplace=True)
        _perf.columns.name = None
        if 'auc' in _perf.columns:
            _perf.sort_values('auc', inplace=True, ascending=False)
        else:
            _perf.sort_values('mse', inplace=True, ascending=True)
        print(_perf.to_string())
        if opts.out_summary:
            perf.to_csv(opts.out_summary, sep='\t', index=False)

        if opts.out_data:
            dat.h5_write_data(data, opts.out_data)

        log.info('Done!')

        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
