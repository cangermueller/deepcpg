#!/usr/bin/env python

import sys
import os

import argparse
import logging

from deepcpg import data as dat
from deepcpg import evaluation as ev
from deepcpg import models as mod
from deepcpg.data import hdf
from deepcpg.utils import ProgressBar, to_list


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

        log.info('Loading data ...')
        nb_sample = dat.get_nb_sample(opts.data_files, opts.nb_sample)
        data_reader = mod.data_reader_from_model(model)

        data_reader = data_reader(opts.data_files,
                                  nb_sample=nb_sample,
                                  batch_size=opts.batch_size,
                                  loop=False, shuffle=False)

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

        eval_report = ev.evaluate_outputs(data['outputs'], data['preds'])

        if opts.out_summary:
            eval_report.to_csv(opts.out_summary, sep='\t', index=False)

        eval_report = ev.unstack_report(eval_report)
        print(eval_report.to_string())

        if opts.out_data:
            hdf.write_data(data, opts.out_data)

        log.info('Done!')

        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
