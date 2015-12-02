#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import pandas as pd
import numpy as np
import h5py as h5
import yaml
import random
from keras.callbacks import EarlyStopping, ModelCheckpoint

from predict.models.dnn.utils import evaluate_all, load_model, MASK, open_hdf, read_labels
from predict.models.dnn.utils import write_z, map_targets, ArrayView
from predict.models.dnn.callbacks import LearningRateScheduler, PerformanceLogger, ProgressLogger
from predict.models.dnn.callbacks import DataJumper
import predict.models.dnn.model as mod



def sample_weights(y, weight_classes=False):
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
            description='Build and train model')
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
            help='Restart training from JSON, weights file',
            nargs=2)
        p.add_argument(
            '--model_weights',
            help='Model weights')
        p.add_argument(
            '--max_mem',
            help='Maximum memory load',
            type=int,
            default=14000)
        p.add_argument(
            '--nb_sample',
            help='Maximum # training samples',
            type=int)
        p.add_argument(
            '--nb_epoch',
            help='Maximum # training epochs',
            type=int)
        p.add_argument(
            '--weight_classes',
            help='Weight classes by frequency',
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

        pd.set_option('display.width', 150)

        labels = read_labels(opts.train_file)
        targets = labels['targets']

        f = h5.File(opts.train_file)
        seq_len = f['/data/s_x'].shape[1]
        cpg_len = f['/data/c_x'].shape[3]
        nb_unit = f['/data/c_x'].shape[2]
        f.close()

        model_params = mod.Params()
        if opts.model_params:
            with open(opts.model_params, 'r') as f:
                configs = yaml.load(f.read())
                model_params.update(configs)

        print('Model parameters:')
        print(model_params)
        model_params.to_yaml(pt.join(opts.out_dir, 'model_params.yaml'))
        print()

        if opts.retrain is None:
            log.info('Build model')
            model = mod.build(model_params, targets, seq_len, cpg_len,
                              nb_unit=nb_unit)
        else:
            log.info('Retrain %s' % (opts.retrain[0]))
            model = load_model(opts.retrain[0], opts.retrain[1])
        with open(pt.join(opts.out_dir, 'model.json'), 'w') as f:
            f.write(model.to_json())

        log.info('Fit model')
        model_weights_last = pt.join(opts.out_dir, 'model_weights_last.h5')
        model_weights_best = pt.join(opts.out_dir, 'model_weights_best.h5')

        cb = []
        cb.append(ProgressLogger())
        cb.append(ModelCheckpoint(model_weights_last, save_best_only=False))
        cb.append(ModelCheckpoint(model_weights_best,
                                  save_best_only=True, verbose=1))
        cb.append(EarlyStopping(patience=model_params.early_stop, verbose=1))

        def lr_schedule():
            old_lr = model.optimizer.lr.get_value()
            new_lr = old_lr * model_params.lr_decay
            model.optimizer.lr.set_value(new_lr)
            print('Learning rate dropped from %.4f to %.4f' % (old_lr, new_lr))

        cb.append(LearningRateScheduler(lr_schedule,
                                        patience=model_params.early_stop - 1))
        perf_logger = PerformanceLogger()
        cb.append(perf_logger)

        def read_data(path):
            f = open_hdf(path, cache_size=opts.max_mem)
            data = dict()
            for k, v in f['data'].items():
                data[k] = v
            for k, v in f['pos'].items():
                data[k] = v
            return (f, data)

        weight_classes = opts.weight_classes
        if not weight_classes:
            weight_classes = model_params.weight_classes

        train_file, train_data = read_data(opts.train_file)
        train_weights = dict()
        for k in model.output_order:
            train_weights[k] = sample_weights(train_data[k],
                                              weight_classes)

        if opts.val_file is None:
            val_data = train_data
            val_weights = train_weights
            val_file = None
        else:
            val_file, val_data = read_data(opts.val_file)
            val_weights = dict()
            for k in model.output_order:
                val_weights[k] = sample_weights(val_data[k],
                                                weight_classes)

        def to_view(d):
            for k in d.keys():
                d[k] = ArrayView(d[k])

        to_view(train_data)
        to_view(train_weights)
        views = list(train_data.values()) + list(train_weights.values())
        cb.append(DataJumper(views, nb_sample=opts.nb_sample, verbose=1, jump=True))

        if val_data is not train_data:
            to_view(val_data)
            to_view(val_weights)
            views = list(val_data.values()) + list(val_weights.values())
            cb.append(DataJumper(views, nb_sample=opts.nb_sample, verbose=1,
                                 jump=False))

        print('%d training samples' % (list(train_data.values())[0].shape[0]))
        print('%d validation samples' % (list(val_data.values())[0].shape[0]))

        nb_epoch = opts.nb_epoch
        if nb_epoch is None:
            nb_epoch = model_params.nb_epoch
        model.fit(data=train_data,
                  sample_weight=train_weights,
                  val_data=val_data,
                  val_sample_weight=val_weights,
                  batch_size=model_params.batch_size,
                  shuffle=model_params.shuffle,
                  nb_epoch=nb_epoch,
                  callbacks=cb,
                  verbose=0)

        if pt.isfile(model_weights_best):
            model.load_weights(model_weights_best)

        t = perf_logs_str(perf_logger.frame())
        print('\n\nLearning curve:')
        print(t)
        with open(pt.join(opts.out_dir, 'lc.csv'), 'w') as f:
            f.write(t)

        t = perf_logs_str(perf_logger.batch_frame())
        with open(pt.join(opts.out_dir, 'lc_batch.csv'), 'w') as f:
            f.write(t)

        print('\nValidation set performance:')
        z = model.predict(val_data)
        write_z(val_data, z, labels, pt.join(opts.out_dir, 'z_val.h5'))

        p = evaluate_all(val_data, z)
        loss = model.evaluate(val_data, sample_weight=val_weights)
        p['loss'] = loss
        p.index = map_targets(p.index.values, labels)
        p.index.name = 'target'
        p.reset_index(inplace=True)
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
