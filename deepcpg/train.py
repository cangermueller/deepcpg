#!/usr/bin/env python

import os
import random
import sys

import argparse
import logging
import numpy as np
import pandas as pd

from keras.backend import _BACKEND
from keras.models import Model
from keras.optimizers import Adam

from keras import callbacks as kcbk

from deepcpg import callbacks as cbk
from deepcpg import data as dat
from deepcpg import models as mod
from deepcpg import utils as ut


LOG_PRECISION = 5


def get_class_weights(data_files, output_names):
    names = {'outputs': output_names}
    weights = {}
    for name, output in dat.h5_read(data_files, names).items():
        output = output[output != dat.CPG_NAN]
        frac_ones = np.sum(output == 1) / len(output)
        weights[name.replace('outputs/', '')] = {0: 1 - frac_ones, 1: frac_ones}
    return weights


def perf_logs_str(logs):
    t = logs.to_csv(None, sep='\t', float_format='%.4f', index=False)
    return t


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
            '--output_names',
            help='List of regex to filter outputs',
            nargs='+')
        p.add_argument(
            '--nb_output',
            type=int,
            help='Maximum number of outputs')
        p.add_argument(
            '--replicate_names',
            help='List of regex to filter CpG context units',
            nargs='+')
        p.add_argument(
            '--model_name',
            help='Model name',
            default='dna01')
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
            '--early_stopping',
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
            help='DNA window length',
            type=int)
        p.add_argument(
            '--cpg_wlen',
            help='CpG window length',
            type=int)
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

        if opts.val_files:
            h = kcbk.EarlyStopping('val_loss' if opts.val_files else 'loss',
                                   patience=opts.early_stopping,
                                   verbose=1)
            cbacks.append(h)

        h = kcbk.ModelCheckpoint(os.path.join(opts.out_dir,
                                              'model_weights_last.h5'),
                                 save_best_only=False)
        cbacks.append(h)
        monitor = 'val_loss' if opts.val_files else 'loss'
        h = kcbk.ModelCheckpoint(os.path.join(opts.out_dir, 'model_weights.h5'),
                                 monitor=monitor,
                                 save_best_only=True, verbose=1)
        cbacks.append(h)

        if opts.max_time is not None:
            cbacks.append(cbk.Timer(opts.max_time * 3600 * 0.8))

        h = kcbk.LearningRateScheduler(
            lambda epoch: opts.lr * opts.lr_decay**epoch)
        cbacks.append(h)

        def save_lc(epoch, epoch_logs, val_epoch_logs):
            logs = {'lc.csv': epoch_logs, 'lc_val.csv': val_epoch_logs}
            for name, logs in logs.items():
                if not logs:
                    continue
                logs = pd.DataFrame(logs)
                with open(os.path.join(opts.out_dir, name), 'w') as f:
                    f.write(perf_logs_str(logs))

        self.perf_logger = cbk.PerformanceLogger(callbacks=[save_lc],
                                                 precision=LOG_PRECISION)
        cbacks.append(self.perf_logger)

        if _BACKEND == 'tensorflow':
            h = kcbk.TensorBoard(
                log_dir=os.path.join(opts.out_dir, 'logs'),
                histogram_freq=1,
                write_graph=True,
                write_images=True)
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

        self.log = log
        self.opts = opts

        # Create output directory if not existing
        if not os.path.exists(opts.out_dir):
            os.makedirs(opts.out_dir, exist_ok=True)

        # Setup callbacks
        log.info('Initializing callbacks ...')
        cbacks = self.callbacks()

        log.info('Building model ...')
        output_names = dat.h5_ls(opts.train_files[0], 'outputs',
                                 opts.output_names, opts.nb_output)
        class_weights = get_class_weights(opts.train_files, output_names)

        model_builder = mod.get_class(opts.model_name)

        if opts.model_name.lower().startswith('dna'):
            dna_wlen = dat.get_dna_wlen(opts.train_files[0], opts.dna_wlen)
            model_builder = model_builder(dna_wlen=dna_wlen,
                                          dropout=opts.dropout,
                                          l1_decay=opts.l1_decay,
                                          l2_decay=opts.l2_decay)

        elif opts.model_name.lower().startswith('cpg'):
            cpg_wlen = dat.get_cpg_wlen(opts.train_files[0], opts.cpg_wlen)
            replicate_names = dat.h5_ls(opts.train_files[0], 'inputs/cpg',
                                        opts.replicate_names)
            model_builder = model_builder(replicate_names,
                                          cpg_wlen=cpg_wlen,
                                          dropout=opts.dropout,
                                          l1_decay=opts.l1_decay,
                                          l2_decay=opts.l2_decay)
        else:
            dna_wlen = dat.get_dna_wlen(opts.train_files[0], opts.dna_wlen)
            cpg_wlen = dat.get_cpg_wlen(opts.train_files[0], opts.cpg_wlen)
            replicate_names = dat.h5_ls(opts.train_files[0], 'inputs/cpg',
                                        opts.replicate_names)
            model_builder = model_builder(replicate_names=replicate_names,
                                          cpg_wlen=cpg_wlen,
                                          dna_wlen=dna_wlen,
                                          dropout=opts.dropout,
                                          l1_decay=opts.l1_decay,
                                          l2_decay=opts.l2_decay)

        inputs = model_builder.inputs()
        stem = model_builder(inputs)
        outputs = mod.add_outputs(stem, output_names)
        model = Model(input=inputs, output=outputs, name=opts.model_name)
        model.summary()
        mod.save_model(model, os.path.join(opts.out_dir, 'model.json'))

        optimizer = Adam(lr=opts.lr)
        log.info('Compile model ...')
        model.compile(optimizer=optimizer, loss='binary_crossentropy',
                      metrics=['accuracy'])

        log.info('Reading data ...')
        nb_train_sample = dat.get_nb_sample(opts.train_files,
                                            opts.nb_train_sample,
                                            opts.batch_size)
        train_data = model_builder.reader(opts.train_files,
                                          output_names=output_names,
                                          batch_size=opts.batch_size,
                                          nb_sample=nb_train_sample,
                                          loop=True, shuffle=True,
                                          class_weights=class_weights)
        import ipdb; ipdb.set_trace()
        if opts.val_files:
            nb_val_sample = dat.get_nb_sample(opts.val_files,
                                              opts.nb_val_sample,
                                              opts.batch_size)
            val_data = model_builder.reader(opts.val_files,
                                            output_names=output_names,
                                            batch_size=opts.batch_size,
                                            nb_sample=nb_val_sample,
                                            loop=True, shuffle=False,
                                            class_weights=class_weights)
        else:
            val_data = None
            nb_val_sample = None

        # Train model
        log.info('Training model ...')
        print()
        model.fit_generator(
            train_data, nb_train_sample, opts.nb_epoch,
            callbacks=cbacks,
            validation_data=val_data,
            nb_val_samples=nb_val_sample,
            max_q_size=opts.data_q_size,
            nb_worker=opts.data_nb_worker,
            verbose=1 if opts.verbose else 0)

        # Use best weights on validation set
        h = os.path.join(opts.out_dir, 'model_weights.h5')
        if os.path.isfile(h):
            model.load_weights(h)

        model.save(os.path.join(opts.out_dir, 'model.h5'))

        print('\nTraining set performance:')
        print(ut.format_table(self.perf_logger.epoch_logs, precision=5))

        if self.perf_logger.val_epoch_logs:
            print('\nValidation set performance:')
            print(ut.format_table(self.perf_logger.val_epoch_logs,
                                  precision=LOG_PRECISION))

        log.info('Done!')

        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
