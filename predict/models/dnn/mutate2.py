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
            '--out_group',
            help='Output group')
        p.add_argument(
            '--nb_sample',
            help='Maximum # training samples',
            type=int)
        p.add_argument(
            '--type',
            help='Mutation type',
            choices=['zero', 'shuffle', 'rnd', 'max'],
            default='rnd')
        p.add_argument(
            '--rnd_nb',
            help='Number of repetitions for random mutations',
            type=int,
            default=1)
        p.add_argument(
            '--wlen',
            help='Window length',
            type=int,
            default=0)
        p.add_argument(
            '--exclude_center',
            help='Exclude center position',
            action='store_true')
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

        log.info('Load model')
        #  model = mod.model_from_list(opts.model)

        log.info('Load data')
        targets = ut.read_targets(opts.data_file)
        data_file, data = ut.read_hdf(opts.data_file, opts.max_mem)
        ut.to_view(data, stop=opts.nb_sample)

        print('%d samples' % (list(data.values())[0].shape[0]))
        print()

        def progress(*args, **kwargs):
            h = mod.progress(*args, **kwargs)
            if h is not None:
                print(h)

        def predict_and_write(name='zm'):
            z = model.predict(data, callbacks=[progress],
                              batch_size=opts.batch_size)
            ut.write_z(data, z, targets, opts.out_file, unlabeled=True,
                       name=name)
            return z

        log.info('Predict')
        #  predict_and_write('z')

        seq = data['s_x'][:]
        seq_len = seq.shape[1]

        if opts.type == 'zero':
            if opts.wlen == -1:
                mseq = np.zeros(seq.shape, dtype='int8')
            else:
                mseq = seq.copy()
                mseq[:, slice_center(seq_len, opts.wlen)] = 0
            data['s_x'] = mseq
            #  predict_and_write('zm')

        elif opts.type == 'shuffle':
            if opts.wlen == -1:
                mseq = np.zeros(seq.shape, dtype='int8')
                d = mseq
            else:
                mseq = seq.copy()
                d = mseq[:, slice_center(seq_len, opts.wlen)]
                d[:] = 0
            h = np.random.randint(0, d.shape[2],
                                    (d.shape[0], d.shape[1]))
            for i in range(d.shape[2]):
                d[h == i, i] = 1
            data['s_x'] = mseq
            #  predict_and_write()

        elif opts.type == 'rnd':
            mseq = seq.copy()
            d = mseq[:, slice_center(seq_len, opts.wlen)]
            p = np.random.randint(0, d.shape[1], (d.shape[0], opts.rnd_nb))
            h = np.arange(d.shape[0])
            d[h, p.T] = 0
            assert np.all(d[0, p[0]] == 0)
            dp = d[h, p.T]
            dp = np.swapaxes(dp, 0, 1)
            m = np.random.randint(0, dp.shape[2], dp.shape[:2])
            for i in range(dp.shape[2]):
                dp[m == i, i] = 1
            dp = np.swapaxes(dp, 0, 1)
            d[h, p.T] = dp
            # TODO: remove
            assert np.all(mseq.sum(axis=2) == 1)
            assert np.any(mseq != seq)
            data['s_x'] = mseq
            #  predict_and_write('zm')

        elif opts.type == 'max':
            for i in range(opts.wlen):
                for j in range(seq.shape[2]):
                    mseq = seq.copy()
                    d = mseq[:, slice_center(seq_len, opts.wlen)]
                    d[:, i] = 0
                    d[:, i, j] = 1
                    data['s_x'] = mseq
                    #  predict_and_write('zm')

        data_file.close()

        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
