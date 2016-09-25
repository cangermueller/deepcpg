#!/usr/bin/env python

import argparse
import sys
import logging
import os
import os.path as pt
import random

import h5py as h5
import numpy as np

from keras import backend as kback
from keras import layers as kl
from keras import models as kmod
from keras import optimizers as kopt

from keras import callbacks as kcbk

from deepcpg import callbacks as cbk
from deepcpg.models import base
from deepcpg.data import preprocess as pp


# TODO:
# Normalize inputs


def get_targets(data_file, target_filter=None):
    data_file = h5.File(data_file, 'r')
    targets = list(data_file['cpg'].keys())
    if target_filter is not None:
        if isinstance(target_filter, list):
            targets = [target for target in targets if target in target_filter]
        else:
            targets = targets[:target_filter]
    data_file.close()
    return targets


def perf_logs_str(logs):
    t = logs.to_csv(None, sep='\t', float_format='%.4f', index=False)
    return t


def get_dna_wlen(data_file):
    data_file = h5.File(data_file, 'r')
    wlen = data_file['dna'].shape[1]
    return wlen


def get_cpg_wlen(data_file):
    data_file = h5.File(data_file, 'r')
    wlen = data_file['dna'].shape[1]
    return wlen


def count_samples(data_files, nb_max=None, batch_size=None):
    nb_sample = 0
    for data_file in data_files:
        data_file = h5.File(data_file, 'r')
        nb_sample += len(data_file['pos'])
        data_file.close()
        if nb_max and nb_sample > nb_max:
            nb_sample = nb_max
            break
    if batch_size:
        nb_sample = (nb_sample // batch_size) * batch_size
    return nb_sample


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
            description='Train model')
        p.add_argument(
            'train_files',
            nargs='+',
            help='Training data files')
        p.add_argument(
            '--val_files',
            nargs='+',
            help='Validation data files')
        p.add_argument(
            '-o', '--out_dir',
            default='./train',
            help='Output directory')
        p.add_argument(
            '--targets',
            help='Target names or number of targets',
            nargs='+')
        p.add_argument(
            '--nb_epoch',
            help='Maximum # training epochs',
            type=int,
            default=100)
        p.add_argument(
            '--batch_size',
            help='Batch size',
            type=int,
            default=128)
        p.add_argument(
            '--early_stop',
            help='Early stopping patience',
            type=int,
            default=3)
        p.add_argument(
            '--lr',
            help='Learning rate',
            type=float,
            default=0.0001)
        p.add_argument(
            '--lr_decay',
            help='Exponential learning rate decay factor',
            type=float,
            default=0.95)
        p.add_argument(
            '--nb_train_sample',
            help='Maximum # training samples per epoch',
            type=int)
        p.add_argument(
            '--nb_val_sample',
            help='Maximum # validation samples per epoch',
            type=int)
        p.add_argument(
            '--max_time',
            help='Maximum training time in hours',
            type=float)
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
        p.add_argument(
            '--dna_wlen',
            help='DNA window lengths',
            type=int,
            default=501)
        p.add_argument(
            '--dropout',
            help='Dropout rate',
            type=float,
            default=0.0)
        p.add_argument(
            '--l1_decay',
            help='L1 weight decay',
            type=float,
            default=0.0)
        p.add_argument(
            '--l2_decay',
            help='L2 weight decay',
            type=float,
            default=0.0)
        p.add_argument(
            '--data_q_size',
            help='Size of data generator queue',
            type=int,
            default=10)
        p.add_argument(
            '--data_nb_worker',
            help='Number of worker for data generator queue',
            type=int,
            default=1)
        return p

    def callbacks(self):
        opts = self.opts
        cbacks = []

        h = kcbk.EarlyStopping('val_loss' if opts.val_files else 'loss',
                               patience=opts.early_stop,
                               verbose=1)
        cbacks.append(h)

        h = kcbk.ModelCheckpoint(pt.join(opts.out_dir, 'model_weights_last.h5'),
                                 save_best_only=False)
        cbacks.append(h)
        h = kcbk.ModelCheckpoint(pt.join(opts.out_dir, 'model_weights.h5'),
                                 save_best_only=True, verbose=1)
        cbacks.append(h)

        if opts.max_time is not None:
            cbacks.append(cbk.Timer(opts.max_time * 3600 * 0.8))

        h = kcbk.LearningRateScheduler(
            lambda epoch: opts.lr * opts.lr_decay**epoch)
        cbacks.append(h)

        def save_lc():
            log = {'lc.csv': perf_logger.epoch_frame()}
            for k, v in log.items():
                with open(pt.join(opts.out_dir, k), 'w') as f:
                    f.write(perf_logs_str(v))

        # TODO: Check with val loss; grep loss / acc?
        perf_logger = cbk.PerformanceLogger(epoch_logs='all', batch_logs=None, callbacks=[save_lc])
        cbacks.append(perf_logger)

        if kback._BACKEND == 'tensorflow':
            h = kcbk.TensorBoard(
                log_dir=pt.join(opts.out_dir, 'logs'),
                histogram_freq=1,
                write_graph=True,
                write_images=False)
            cbacks.append(h)

        return cbacks

    def main(self, name, opts):
        logging.basicConfig(filename=opts.log_file,
                            format='%(levelname)s (%(asctime)s): %(message)s')
        log = logging.getLogger(name)
        if opts.verbose:
            log.setLevel(logging.DEBUG)
        else:
            log.setLevel(logging.INFO)

        if opts.seed is not None:
            np.random.seed(opts.seed)
            random.seed(opts.seed)
        sys.setrecursionlimit(10**6)

        self.log = log
        self.opts = opts

        # Create output directory if not existing
        if not pt.exists(opts.out_dir):
            os.makedirs(opts.out_dir, exist_ok=True)

        # Setup callbacks
        log.info('Setup callbacks')
        cbacks = self.callbacks()

        targets = get_targets(opts.train_files[0], opts.targets)

        # TODO: class weights
        nb_train_sample = count_samples(opts.train_files, opts.nb_train_sample, opts.batch_size)
        train_data = base.data_generator(opts.train_files, targets,
                                         dna_wlen=0, cpg_wlen=None,
                                         batch_size=opts.batch_size,
                                         nb_sample=nb_train_sample)
        if opts.val_files:
            nb_val_sample = count_samples(opts.val_files, opts.nb_val_sample, opts.batch_size)
            val_data = base.data_generator(opts.val_files, targets,
                                           dna_wlen=0, cpg_wlen=None,
                                           batch_size=opts.batch_size,
                                           nb_sample=nb_val_sample)
        else:
            val_data = None
            nb_val_sample = None

        log.info('Build model ...')
        inputs = []
        inputs.append(kl.Input(shape=(get_dna_wlen(opts.train_files[0]), 4),
                               name='x/dna'))
        stem = base.model(inputs[0],
                          dropout=opts.dropout,
                          l1_decay=opts.l1_decay,
                          l2_decay=opts.l2_decay)
        outputs = base.add_cpg_outputs(stem, targets)
        model = kmod.Model(input=inputs, output=outputs)
        model.summary()
        base.save_model(model, pt.join(opts.out_dir, 'model.json'))

        optimizer = kopt.Adam(lr=opts.lr)
        log.info('Compile model ...')
        model.compile(optimizer=optimizer, loss='binary_crossentropy',
                      metrics=['accuracy'])

        # Train model
        log.info('Train model')
        model.fit_generator(
            train_data, nb_train_sample, opts.nb_epoch,
            callbacks=cbacks,
            validation_data=val_data,
            nb_val_samples=nb_val_sample,
            max_q_size=opts.data_q_size,
            nb_worker=opts.data_nb_worker,
            verbose=1 if opts.verbose else 2)

        # Use best weights on validation set
        h = pt.join(opts.out_dir, 'model_weights.h5')
        if pt.isfile(h):
            model.load_weights(h)

        model.save(pt.join(opts.out_dir, 'model.h5'))

        if opts.nb_epoch > 0:
            for cback in cbacks:
                if isinstance(cback, cbk.PerformanceLogger):
                    perf_logger = cback
                    break
            lc = perf_logger.epoch_frame()
            print('\n\nLearning curve:')
            print(perf_logs_str(lc))
            if len(lc) > 5:
                lc = lc.loc[lc.epoch > 2]
            lc.set_index('epoch', inplace=True)
            ax = lc.plot(figsize=(10, 6))
            ax.get_figure().savefig(pt.join(opts.out_dir, 'lc.png'))

        log.info('Done!')

        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
