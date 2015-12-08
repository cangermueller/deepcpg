#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import pandas as pd
import numpy as np
import h5py as h5
import random
import pickle
from keras.callbacks import EarlyStopping, ModelCheckpoint

from predict.models.dnn.utils import evaluate_all, load_model, MASK, open_hdf, read_labels
from predict.models.dnn.utils import write_z, map_targets, ArrayView
from predict.models.dnn.callbacks import DataJumper
from predict.models.dnn.callbacks import LearningRateScheduler, PerformanceLogger, ProgressLogger, Timer
import predict.models.dnn.model as mod


def get_sample_weights(y, weight_classes=False):
    y = y[:]
    class_weights = {-1: 0, 0: 1, 1: 1}
    if weight_classes:
        t = y[y != MASK].mean()
        class_weights[0] = t
        class_weights[1] = 1 - t
    sample_weights = np.zeros(y.shape, dtype='float16')
    for k, v in class_weights.items():
        sample_weights[y == k] = v
    return sample_weights


def perf_logs_str(logs):
    t = logs.to_csv(None, sep='\t', float_format='%.4f', index=False)
    return t


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
            'train_file',
            help='Training data file')
        p.add_argument(
            '--val_file',
            help='Validation data file')
        p.add_argument(
            '-o', '--out_dir',
            help='Output directory')
        p.add_argument(
            '--model_params',
            help='Model parameters file')
        p.add_argument(
            '--retrain',
            help='Restart training from pickled or json/weights file',
            nargs='+')
        p.add_argument(
            '--nb_epoch',
            help='Maximum # training epochs',
            type=int,
            default=100)
        p.add_argument(
            '--batch_size',
            help='Batch size',
            type=int)
        p.add_argument(
            '--early_stop',
            help='Early stopping patience',
            type=int,
            default=3)
        p.add_argument(
            '--lr_decay',
            help='Learning schedule decay rate',
            type=float,
            default=0.5)
        p.add_argument(
            '--shuffle',
            help='Data shuffling',
            default='batch')
        p.add_argument(
            '--weight_classes',
            help='Weight classes by frequency',
            action='store_true')
        p.add_argument(
            '--nb_sample',
            help='Maximum # training samples per epoch',
            type=int)
        p.add_argument(
            '--max_time',
            help='Maximum training time in hours',
            type=float)
        p.add_argument(
            '--max_mem',
            help='Maximum memory load',
            type=int,
            default=14000)
        p.add_argument(
            '--no_jump',
            help='Do not jump in training set',
            action='store_true')
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

    def adjust_batch_size(self, model, data, sample_weights):
        configs = [
            (1024, 0.0001),
            (768, 0.000125),
            (512, 0.00025),
            (256, 0.0005),
            (128, 0.001),
            (64, 0.002),
            (32, 0.004)
        ]
        idx = None
        for i in range(len(configs)):
            batch_size = configs[i][0]
            self.log.info('Try batch size %d' % (batch_size))
            batch_data = {k: v[:batch_size] for k, v in data.items()}
            batch_weights = {k: v[:batch_size] for k, v in sample_weights.items()}
            try:
                model.train_on_batch(batch_data,
                                     sample_weight=batch_weights)
                idx = i
                break
            except:
                self.log.info('Batch size %d failed!' % (batch_size))
        if idx is None:
            return (None, None)
        else:
            return (configs[idx])

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
            random.seed(opts.seed)
        sys.setrecursionlimit(10**6)

        pd.set_option('display.width', 150)
        self.log = log
        self.opts = opts

        # Initialize variables
        labels = read_labels(opts.train_file)
        targets = labels['targets']

        f = h5.File(opts.train_file)
        seq_len = f['/data/s_x'].shape[1]
        cpg_len = f['/data/c_x'].shape[3]
        nb_unit = f['/data/c_x'].shape[2]
        f.close()

        # Setup model
        if opts.model_params:
            model_params = mod.Params.from_yaml(opts.model_params)

        if opts.retrain is None:
            log.info('Build model')
            if model_params is None:
                model_params = mod.Params()
            model = mod.build(model_params, targets, seq_len, cpg_len,
                              nb_unit=nb_unit)
            log.info('Pickle model')
            with open(pt.join(opts.out_dir, 'model.pkl'), 'wb') as f:
                pickle.dump(model, f)
        else:
            log.info('Retrain %s' % (opts.retrain[0]))
            if len(opts.retrain) == 2:
                model = load_model(opts.retrain[0], opts.retrain[1])
            else:
                with open(opts.retrain[0], 'rb') as f:
                    model = pickle.load(f)

        with open(pt.join(opts.out_dir, 'model.json'), 'w') as f:
            f.write(model.to_json())

        if model_params is not None:
            print('Model parameters:')
            print(model_params)
            model_params.to_yaml(pt.join(opts.out_dir, 'model_params.yaml'))
            print()

        log.info('Setup training')

        model_weights_last = pt.join(opts.out_dir, 'model_weights_last.h5')
        model_weights_best = pt.join(opts.out_dir, 'model_weights_best.h5')

        cb = []
        cb.append(ProgressLogger())
        cb.append(ModelCheckpoint(model_weights_last, save_best_only=False))
        cb.append(ModelCheckpoint(model_weights_best,
                                  save_best_only=True, verbose=1))
        cb.append(EarlyStopping(patience=opts.early_stop, verbose=1))

        def lr_schedule():
            old_lr = model.optimizer.lr.get_value()
            new_lr = old_lr * opts.lr_decay
            model.optimizer.lr.set_value(new_lr)
            print('Learning rate dropped from %.4f to %.4f' % (old_lr, new_lr))

        cb.append(LearningRateScheduler(lr_schedule,
                                        patience=opts.early_stop - 1))

        if opts.max_time is not None:
            cb.append(Timer(opts.max_time * 3600 * 0.8))

        def save_lc():
            log = {'lc.csv': perf_logger.frame(),
                   'lc_batch.csv': perf_logger.batch_frame()}
            for k, v in log.items():
                with open(pt.join(opts.out_dir, k), 'w') as f:
                    f.write(perf_logs_str(v))

        perf_logger = PerformanceLogger(callbacks=[save_lc])
        cb.append(perf_logger)

        def read_data(path):
            f = open_hdf(path, cache_size=opts.max_mem)
            data = dict()
            for k, v in f['data'].items():
                data[k] = v
            for k, v in f['pos'].items():
                data[k] = v
            return (f, data)


        train_file, train_data = read_data(opts.train_file)
        train_weights = dict()
        for k in model.output_order:
            train_weights[k] = get_sample_weights(train_data[k],
                                                  opts.weight_classes)

        if opts.val_file is None:
            val_data = train_data
            val_weights = train_weights
            val_file = None
        else:
            val_file, val_data = read_data(opts.val_file)
            val_weights = dict()
            for k in model.output_order:
                val_weights[k] = get_sample_weights(val_data[k],
                                                    opts.weight_classes)

        def to_view(d):
            for k in d.keys():
                d[k] = ArrayView(d[k])

        to_view(train_data)
        to_view(train_weights)
        views = list(train_data.values()) + list(train_weights.values())
        cb.append(DataJumper(views, nb_sample=opts.nb_sample, verbose=1,
                             jump=not opts.no_jump))

        if val_data is not train_data:
            to_view(val_data)
            to_view(val_weights)
            views = list(val_data.values()) + list(val_weights.values())
            cb.append(DataJumper(views, nb_sample=opts.nb_sample, verbose=1,
                                 jump=False))

        print('%d training samples' % (list(train_data.values())[0].shape[0]))
        print('%d validation samples' % (list(val_data.values())[0].shape[0]))

        batch_size = opts.batch_size
        if batch_size is None:
            log.info('Adjust batch size')
            batch_size, lr = self.adjust_batch_size(model,
                                                    train_data,
                                                    train_weights)
            if batch_size is None:
                log.error('GPU memory to small')
                return 1
            log.info('Use batch size %d' % (batch_size))
            log.info('Use batch lr %f' % (lr))
            model.optimizer.lr.set_value(lr)

        def logger(x):
            log.debug(x)

        log.info('Train model')
        model.fit(data=train_data,
                  sample_weight=train_weights,
                  val_data=val_data,
                  val_sample_weight=val_weights,
                  batch_size=batch_size,
                  shuffle=opts.shuffle,
                  nb_epoch=opts.nb_epoch,
                  callbacks=cb,
                  verbose=0,
                  logger=logger)

        # Use best weights on validation set
        if pt.isfile(model_weights_best):
            model.load_weights(model_weights_best)

        log.info('Pickle model')
        with open(pt.join(opts.out_dir, 'model.pkl'), 'wb') as f:
            pickle.dump(model, f)

        print('\n\nLearning curve:')
        print(perf_logs_str(perf_logger.frame()))

        print('\nValidation set performance:')
        z = model.predict(val_data, batch_size=batch_size)
        write_z(val_data, z, labels, pt.join(opts.out_dir, 'z_val.h5'))

        p = evaluate_all(val_data, z)
        loss = model.evaluate(val_data, sample_weight=val_weights)
        p['loss'] = loss
        p.index = map_targets(p.index.values, labels)
        p.index.name = 'target'
        p.reset_index(inplace=True)
        p.sort_values('target', inplace=True)
        print(p.to_string(index=False))
        with open(pt.join(opts.out_dir, 'perf_val.csv'), 'w') as f:
            f.write(p.to_csv(None, sep='\t', index=False))

        train_file.close()
        if val_file:
            val_file.close()

        log.info('Done!')

        return 0


if __name__ == '__main__':
    app=App()
    app.run(sys.argv)
