#!/usr/bin/env python

import sys
import os

import argparse
import pandas as pd
import logging

from deepcpg import data as dat
from deepcpg import evaluation as ev
from deepcpg import models as mod
from deepcpg.data import hdf
from deepcpg.utils import ProgressBar, to_list


def unstack_report(report):
    index = list(report.columns[~report.columns.isin(['metric', 'value'])])
    report = pd.pivot_table(report, index=index, columns='metric',
                            values='value')
    report.reset_index(index, inplace=True)
    report.columns.name = None

    # Sort columns
    columns = list(report.columns)
    sorted_columns = []
    for fun in ev.CAT_METRICS + ev.CLA_METRICS + ev.REG_METRICS:
        for i, column in enumerate(columns):
            if column.startswith(fun.__name__):
                sorted_columns.append(column)
    sorted_columns = index + sorted_columns
    sorted_columns += [col for col in columns if col not in sorted_columns]
    report = report[sorted_columns]
    order = []
    if 'dset' in report.columns:
        order.append(('dset', True))
    if 'auc' in report.columns:
        order.append(('auc', False))
    elif 'mse' in report.columns:
        order.append(('mse', True))
    elif 'acc' in report.columns:
        order.append(('acc', False))
    report.sort_values([x[0] for x in order],
                       ascending=[x[1] for x in order],
                       inplace=True)
    return report


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
            help='Test data files',
            nargs='+')
        p.add_argument(
            '--model_files',
            help='Model files',
            nargs='+')
        p.add_argument(
            '-o', '--out_summary',
            help='Output summary file')
        p.add_argument(
            '--out_data',
            help='Output file with predictions and labels')
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
        data_reader = mod.data_reader_from_model(model)
        data_reader = data_reader(opts.data_files,
                                  nb_sample=nb_sample,
                                  batch_size=opts.batch_size,
                                  loop=False,
                                  shuffle=False)

        meta_reader = hdf.reader(opts.data_files, ['chromo', 'pos'],
                                 nb_sample=nb_sample,
                                 batch_size=opts.batch_size,
                                 loop=False, shuffle=False)

        log.info('Predicting ...')
        data = dict()
        progbar = ProgressBar(nb_sample, log.info)
        for inputs, outputs, weights in data_reader:
            batch_size = len(list(inputs.values())[0])
            progbar.update(batch_size)

            preds = to_list(model.predict(inputs))

            data_batch = dict()
            data_batch['preds'] = dict()
            data_batch['outputs'] = dict()
            for i, name in enumerate(model.output_names):
                data_batch['preds'][name] = preds[i].squeeze()
                data_batch['outputs'][name] = outputs[name].squeeze()

            for name, value in next(meta_reader).items():
                data_batch[name] = value
            dat.add_to_dict(data_batch, data)
        progbar.close()
        data = dat.stack_dict(data)

        perf = []
        for output in model.output_names:
            if output in ['stats/cat_var']:
                tmp = ev.evaluate_cat(data['outputs'][output],
                                      data['preds'][output],
                                      binary_metrics=[ev.auc])
            else:
                metrics = mod.get_eval_metrics(output)
                tmp = ev.evaluate(data['outputs'][output],
                                  data['preds'][output],
                                  metrics=metrics)
            tmp = pd.DataFrame({'output': output,
                                'metric': list(tmp.keys()),
                                'value': list(tmp.values())})
            perf.append(tmp)
        perf = pd.concat(perf)
        perf = perf[['metric', 'output', 'value']]
        perf.sort_values(['metric', 'value'], inplace=True)

        if opts.out_summary:
            perf.to_csv(opts.out_summary, sep='\t', index=False)

        report = unstack_report(perf)
        print(report.to_string())

        if opts.out_data:
            hdf.write_data(data, opts.out_data)

        log.info('Done!')

        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
