#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import numpy as np

import predict.models.dnn.utils as ut
import predict.models.dnn.model as mod


def slice_center(n, wlen):
    c = n // 2
    delta = wlen // 2
    return slice(c - delta, c + delta + 1)


class App(object):

    def run(self, args):
        name = pt.basename(args[0])
        parser = self.create_parser(name)
        opts = parser.parse_args(args[1:])
        self.opts = opts
        return self.main(name, opts)

    def create_parser(self, name):
        p = argparse.ArgumentParser(
            prog=name,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Make prediction after mutating DNA sequence')
        p.add_argument(
            'data_file',
            help='Data file')
        p.add_argument(
            '--model',
            help='Model files',
            nargs='+')
        p.add_argument(
            '-o', '--out_file',
            help='Output file')
        p.add_argument(
            '--chromo',
            help='Chromosome')
        p.add_argument(
            '--start',
            help='Start position',
            type=int)
        p.add_argument(
            '--end',
            help='End position',
            type=int)
        p.add_argument(
            '--nb_sample',
            help='Maximum # training samples',
            type=int)
        p.add_argument(
            '--zero_wlen',
            help='Window length for zeroing out sequence',
            nargs='+',
            type=int)
        p.add_argument(
            '--rnd_wlen',
            help='Window length for randomizing sequence',
            nargs='+',
            type=int)
        p.add_argument(
            '--batch_size',
            help='Batch size',
            type=int,
            default=128)
        p.add_argument(
            '--max_mem',
            help='Maximum memory load',
            type=int,
            default=14000)
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

        log.info('Load data')
        data_file, data = ut.read_data(opts.data_file, opts.max_mem)
        labels = ut.read_labels(opts.data_file)
        if opts.chromo is not None:
            if ut.select_data(data, opts.chromo, opts.start, opts.end, log) == 0:
                log.info('No samples satisfy filter')
                return 0

        def to_view(d):
            for k in d.keys():
                d[k] = ut.ArrayView(d[k], stop=opts.nb_sample)

        to_view(data)
        log.info('%d samples' % (list(data.values())[0].shape[0]))

        log.info('Load model')
        model = mod.model_from_list(opts.model)

        def progress(batch, nb_batch):
            batch += 1
            c = max(1, int(np.ceil(nb_batch / 50)))
            if batch == 1 or batch == nb_batch or batch % c == 0:
                print('%5d / %d (%.1f%%)' % (batch, nb_batch,
                                             batch / nb_batch * 100))

        def predict_and_write(name='z'):
            z = model.predict(data, callbacks=[progress],
                              batch_size=opts.batch_size)
            ut.write_z(data, z, labels, opts.out_file, unlabeled=True,
                       name=name)
            return z

        log.info('Predict')
        predict_and_write()

        seq = data['s_x'][:]
        seq_len = seq.shape[1]

        if opts.zero_wlen is not None:
            for wlen in opts.zero_wlen:
                log.info('Zeroing out wlen=%d' % (wlen))
                if wlen == -1:
                    mseq = np.zeros(seq.shape, dtype='int8')
                else:
                    mseq = seq.copy()
                    mseq[:, slice_center(seq_len, wlen)] = 0
                data['s_x'] = mseq
                predict_and_write('z_zero%d' % (wlen))

        if opts.rnd_wlen is not None:
            for wlen in opts.rnd_wlen:
                log.info('Randomize wlen=%d' % (wlen))
                if wlen == -1:
                    mseq = np.zeros(seq.shape, dtype='int8')
                    d = mseq
                else:
                    mseq = seq.copy()
                    d = mseq[:, slice_center(seq_len, wlen)]
                    d[:] = 0
                h = np.random.randint(0, d.shape[2],
                                      (d.shape[0], d.shape[1]))
                for i in range(d.shape[2]):
                    d[h == i, i] = 1
                data['s_x'] = mseq
                predict_and_write('z_rnd%d' % (wlen))

        data['s_x'] = seq
        data_file.close()
        log.info('Done!')

        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
