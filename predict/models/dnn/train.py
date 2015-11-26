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

from predict.evaluation import eval_to_str
from utils import evaluate_all, load_model, MASK, open_hdf
from callbacks import LearningRateScheduler, PerformanceLogger, ProgressLogger
import model as mod



def sample_weights(y, outputs):
    w = dict()
    for o in outputs:
        yo = y[o]
        wo = np.zeros(yo.shape[0], dtype='float32')
        wo[yo[:] != MASK] = 1
        w[o] = wo
    return w

def perf_logs_str(logs):
    t = logs.to_csv(None, sep='\t', float_format='%.3f', index=False)
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
            '--model_file',
            help='JSON model description')
        p.add_argument(
            '--model_weights',
            help='Model weights')
        p.add_argument(
            '--max_mem',
            help='Maximum memory load',
            type=int,
            default=14000)
        p.add_argument(
            '--max_samples',
            help='Maximum # samples',
            type=int)
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

        f = h5.File(opts.train_file)
        targets = [x.decode() for x in f['/labels/targets']]
        seq_len = f['/data/s_x'].shape[1]
        cpg_len = f['/data/c_x'].shape[3]
        f.close()

        model_params = mod.Params()
        if opts.model_params:
            with open(opts.model_params, 'r') as f:
                configs = yaml.load(f.read())
                model_params.update(configs)

        print('Model parameters:')
        print(model_params)
        print()

        if opts.model_file:
            log.info('Build model from file')
            model = load_model(opts.model_file)
        else:
            log.info('Build model')
            model = mod.build(model_params, targets, seq_len, cpg_len)
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
            g = f['data']
            d = {k: g[k] for k in g.keys()}
            if opts.max_samples:
                for k in d.keys():
                    d[k] = d[k][:opts.max_samples]
            return (f, d)

        train_file, train_data = read_data(opts.train_file)
        if opts.val_file is None:
            val_data = train_data
            val_file = None
        else:
            val_file, val_data = read_data(opts.val_file)

        sample_weights_train = sample_weights(train_data, model.output_order)
        sample_weights_val = sample_weights(val_data, model.output_order)

        model.fit(train_data, validation_data=val_data,
                  batch_size=model_params.batch_size,
                  shuffle=model_params.shuffle,
                  nb_epoch=model_params.max_epochs,
                  callbacks=cb,
                  verbose=0,
                  sample_weight=sample_weights_train,
                  sample_weight_val=sample_weights_val)

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
        p = evaluate_all(val_data, z)
        t = eval_to_str(p)
        print(t)
        with open(pt.join(opts.out_dir, 'perf_val.csv'), 'w') as f:
            f.write(t)

        train_file.close()
        if val_file:
            val_file.close()

        log.info('Done!')

        return 0


if __name__ == '__main__':
    app=App()
    app.run(sys.argv)
