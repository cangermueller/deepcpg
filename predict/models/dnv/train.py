#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import pandas as pd
import numpy as np
import h5py as h5
import random
import json
from keras.callbacks import ModelCheckpoint

import predict.evaluation as pe
import predict.models.dnn.utils as ut
import predict.models.dnn.callbacks as cbk
import predict.models.dnv.model as mod
from predict.models.dnv.params import Params


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
            '--targets',
            help='Target names',
            nargs='+')
        p.add_argument(
            '--params',
            help='Model parameters file')
        p.add_argument(
            '--model',
            help='Reuse model',
            nargs='+')
        p.add_argument(
            '--seq_model',
            help='Reuse weights of seq module',
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
            '--max_batch_size',
            help='Maximum batch size',
            type=int,
            default=512)
        p.add_argument(
            '--early_stop',
            help='Early stopping patience',
            type=int,
            default=3)
        p.add_argument(
            '--lr_schedule',
            help='Learning rate scheduler patience',
            type=int,
            default=1)
        p.add_argument(
            '--lr_decay',
            help='Learning schedule decay rate',
            type=float,
            default=0.5)
        p.add_argument(
            '--not_trainable',
            help='Do not train modules starting with letter',
            choices=['s'],
            nargs='+')
        p.add_argument(
            '--shuffle',
            help='Data shuffling',
            default='batch')
        p.add_argument(
            '--nb_sample',
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

    def adjust_batch_size(self, model, data, max_batch_size=None):
        configs = [
            (1024, 0.0001),
            (768, 0.000125),
            (512, 0.000125),
            (256, 0.00025),
            (128, 0.0005),
            (64, 0.001),
            (32, 0.001),
            (16, 0.001)
        ]
        idx = None
        nb_sample = list(data.values())[0].shape[0]
        if max_batch_size is None:
            max_batch_size = nb_sample
        else:
            max_batch_size = min(max_batch_size, nb_sample)
        for i in range(len(configs)):
            batch_size = configs[i][0]
            if batch_size > max_batch_size:
                continue
            self.log.info('Try batch size %d' % (batch_size))
            batch_data = {k: v[:batch_size] for k, v in data.items()}
            try:
                model.train_on_batch(batch_data)
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
        labels = ut.read_labels(opts.train_file)
        if opts.targets is not None:
            idx = []
            for i, x in enumerate(labels['files']):
                if x in opts.targets:
                    idx.append(i)
            for k, v in labels.items():
                labels[k] = [v[i] for i in idx]
        targets = labels['targets']

        h = []
        for f, t in zip(labels['files'], labels['targets']):
            h.append('%s (%s)' % (f, t))
        h = ', '.join(h)
        print('Targets: %s' % (h))

        f = h5.File(opts.train_file)
        seq_len = f['/data/s_x'].shape[1]
        f.close()

        # Setup model
        if opts.params is not None:
            model_params = Params.from_yaml(opts.params)
        else:
            model_params = None

        if opts.model is None:
            log.info('Build model')
            if model_params is None:
                model_params = Params()
            model = mod.build(model_params, targets, seq_len, compile=False)
        else:
            log.info('Loading model')
            model = mod.model_from_list(opts.model, compile=False)

        if opts.seq_model is not None:
            log.info('Copy seq weights')
            seq_model = mod.model_from_list(opts.seq_model, compile=False)
            t = mod.copy_weights(seq_model, model, 's_')
            log.info('Copied weight from %d nodes' % (t))

        if opts.not_trainable is not None:
            for k, v in model.nodes.items():
                if k[0] in opts.not_trainable:
                    log.info("Won't train %s" % (k))
                    v.trainable = False

        log.info('Compile model')
        if model_params is None:
            with open(opts.model[0]) as f:
                config = json.loads(f.read())
            optimizer = mod.optimizer_from_config(config)
        else:
            optimizer = mod.optimizer_from_params(model_params)
        loss = {x: model_params.objective for x in model.output_order}
        model.compile(loss=loss, optimizer=optimizer)

        log.info('Save model')
        mod.model_to_pickle(model, pt.join(opts.out_dir, 'model.pkl'))
        mod.model_to_json(model, pt.join(opts.out_dir, 'model.json'))

        if model_params is not None:
            print('Model parameters:')
            print(model_params)
            h = pt.join(opts.out_dir, 'model_params.yaml')
            if not pt.exists(h):
                model_params.to_yaml(h)
            print()

        log.info('Setup training')

        model_weights_last = pt.join(opts.out_dir, 'model_weights_last.h5')
        model_weights_best = pt.join(opts.out_dir, 'model_weights.h5')

        cb = []
        cb.append(cbk.ProgressLogger())
        cb.append(ModelCheckpoint(model_weights_last,
                                  save_best_only=False))
        cb.append(ModelCheckpoint(model_weights_best,
                                  save_best_only=True, verbose=1))
        cb.append(cbk.EarlyStopping(patience=opts.early_stop, verbose=1))

        def lr_schedule():
            old_lr = model.optimizer.lr.get_value()
            new_lr = old_lr * opts.lr_decay
            model.optimizer.lr.set_value(new_lr)
            print('Learning rate dropped from %g to %g' % (old_lr, new_lr))

        cb.append(cbk.LearningRateScheduler(lr_schedule,
                                            patience=opts.lr_schedule))

        if opts.max_time is not None:
            cb.append(cbk.Timer(opts.max_time * 3600 * 0.8))

        def save_lc():
            log = {'lc.csv': perf_logger.frame(),
                   'lc_batch.csv': perf_logger.batch_frame()}
            for k, v in log.items():
                with open(pt.join(opts.out_dir, k), 'w') as f:
                    f.write(perf_logs_str(v))

        perf_logger = cbk.PerformanceLogger(callbacks=[save_lc])
        cb.append(perf_logger)

        def read_data(path):
            f = ut.open_hdf(path, cache_size=opts.max_mem)
            data = dict()
            for k, v in f['data'].items():
                data[k] = v
            for k, v in f['pos'].items():
                data[k] = v
            return (f, data)

        train_file, train_data = read_data(opts.train_file)

        if opts.val_file is None:
            val_data = train_data
            val_file = None
        else:
            val_file, val_data = read_data(opts.val_file)

        def to_view(d):
            for k in d.keys():
                d[k] = ut.ArrayView(d[k])

        to_view(train_data)
        views = list(train_data.values())
        cb.append(cbk.DataJumper(views, nb_sample=opts.nb_sample, verbose=1,
                                 jump=not opts.no_jump))

        if val_data is not train_data:
            to_view(val_data)
            views = list(val_data.values())
            nb_sample = opts.nb_val_sample
            if nb_sample is None:
                nb_sample = opts.nb_sample
            cb.append(cbk.DataJumper(views,
                                     nb_sample=nb_sample, verbose=1,
                                     jump=False))

        print('%d training samples' % (list(train_data.values())[0].shape[0]))
        print('%d validation samples' % (list(val_data.values())[0].shape[0]))

        batch_size = opts.batch_size
        if batch_size is None:
            log.info('Adjust batch size')
            batch_size, lr = self.adjust_batch_size(
                model, train_data,
                max_batch_size=opts.max_batch_size)
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
                  val_data=val_data,
                  batch_size=batch_size,
                  shuffle=opts.shuffle,
                  nb_epoch=opts.nb_epoch,
                  callbacks=cb,
                  verbose=0,
                  logger=logger)

        # Use best weights on validation set
        if pt.isfile(model_weights_best):
            model.load_weights(model_weights_best)

        log.info('Save model')
        mod.model_to_pickle(model, pt.join(opts.out_dir, 'model.pkl'))

        print('\n\nLearning curve:')
        print(perf_logs_str(perf_logger.frame()))

        print('\nValidation set performance:')
        z = model.predict(val_data, batch_size=batch_size)
        ut.write_z(val_data, z, labels, pt.join(opts.out_dir, 'z_val.h5'))
        p = pe.evaluate_all(val_data, z, mask=None, funs=pe.eval_funs_regress)
        loss = model.evaluate(val_data)
        p['loss'] = loss
        p.index = ut.map_targets(p.index.values, labels)
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
    app = App()
    app.run(sys.argv)
