#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt

import h5py as h5
import numpy as np
import pandas as pd

from deepcpg.models import base
from deepcpg import evaluation as ev



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
            description='Tests the performance of a model')
        p.add_argument(
            'test_files',
            nargs='+',
            help='Test data files')
        p.add_argument(
            'model_files',
            nargs='+',
            help='Model files')
        p.add_argument(
            '-o', '--out_summary',
            help='Output summary file')
        p.add_argument(
            '--out_data',
            help='Output file with predictions and labels')
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

        model = base.load_model(opts.model_files)



        names = dict.fromkeys(['chromo', 'pos'])
        names['inputs'] = model.input_names
        names['outputs'] = model.output_names

        reader = base.hdf5_reader(opts.test_files, names,
                                  batch_size=opts.batch_size,
                                  nb_sample=opts.nb_sample)

        data = dict()
        for data_batch in reader:
            tmp = model.predict(data_batch['inputs'])
            tmp = {name: tmp[i] for i, name in enumerate(model.output_names)}
            data_batch['preds'] = tmp

            for key in ['chromo', 'pos', 'outputs', 'preds']:
                value = data.setdefault(key, [])
                value.append(data_batch[key])

        data = stack_data(data)

        perf = []
        for output in model.output_names:
            tmp = ev.evaluate(data['outputs'][output], data['preds'][output])
            perf.append(pd.DataFrame(tmp, index=[output]))
        perf = pd.concat(perf)
        mean = pd.DataFrame(perf.mean(), index=['mean'])
        perf = pd.concat((perf, mean))
        perf.reset_index(inplace=True)

        tmp = perf.to_csv(None, sep='\t', index=False)
        print(tmp)

        if opts.out_summary:
            with open(opts.out_summary, 'w') as f:
                f.write(tmp)

        if opts.out_data:
            write_data(data, opts.out_data)

        log.info('Done!')

        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
