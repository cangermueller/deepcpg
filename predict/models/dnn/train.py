#!/usr/bin/env python

import argparse
import sys
import logging
import os
import os.path as pt
import pandas as pd
import numpy as np
import h5py as h5
import random
import re
from keras.callbacks import ModelCheckpoint

import predict.evaluation as pe
import predict.utils as pu
import predict.models.dnn.utils as ut
import predict.models.dnn.callbacks as cbk
import predict.models.dnn.model as mod
from predict.models.dnn.params import Params


def get_sample_weights(y, weight_classes=False):
    y = y[:]
    class_weights = {-1: 0, 0: 1, 1: 1}
    if weight_classes:
        t = y[y != ut.MASK].mean()
        class_weights[0] = t
        class_weights[1] = 1 - t
    sample_weights = np.zeros(y.shape, dtype='float16')
    for k, v in class_weights.items():
        sample_weights[y == k] = v
    return sample_weights


def check_weights(model, y, weights):
    for output in model.output_order:
        h = y[output][:] == ut.MASK
        w = weights[output][:]
        assert np.all(w[h] == 0)
        assert np.all(w[~h] == 1)


def perf_logs_str(logs):
    t = logs.to_csv(None, sep='\t', float_format='%.4f', index=False)
    return t


def evaluate(y, z, targets, *args, **kwargs):
    p = pe.evaluate_all(y, z, *args, **kwargs)
    p.index = ut.target_id2name(p.index.values, targets)
    p.index.name = 'target'
    p.reset_index(inplace=True)
    p.sort_values('target', inplace=True)
    return p


def eval_io(model, y, z, out_base, targets):
    if np.any(np.isnan(list(z.values())[0])):
        return None
    ut.write_z(y, z, targets, '%s_z.h5' % (out_base))
    cla = []
    reg = []
    for k, v in model.loss.items():
        if re.match('mse', v):
            reg.append(k)
        else:
            cla.append(k)
    p = []
    if len(cla):
        ys = {k: y[k] for k in cla}
        zs = {k: z[k] for k in cla}
        e = evaluate(ys, zs, targets, funs=pe.eval_funs, mask=ut.MASK)
        if e is not None:
            print('Classification:')
            print(e.to_string(index=False))
            pe.eval_to_file(e, '%s_cla.csv' % (out_base))
        p.append(e)
    else:
        p.append(None)

    if len(reg):
        ys = {k: y[k] for k in cla}
        zs = {k: z[k] for k in cla}
        e = evaluate(ys, zs, targets, funs=pe.eval_funs_regress)
        if e is not None:
            print('Regression:')
            print(e.to_string(index=False))
            pe.eval_to_file(e, '%s_reg.csv' % (out_base))
        p.append(e)
    else:
        p.append(None)


def build_model(params, data_file, targets):
    seq_len = None
    cpg_len = None
    nb_unit = None
    f = h5.File(data_file, 'r')
    g = f['data']
    if 's_x' in g:
        seq_len = g['s_x'].shape[1]
    if 'c_x' in g:
        nb_unit = g['c_x'].shape[2]
        cpg_len = g['c_x'].shape[3]
    f.close()
    model = mod.build(params, targets, seq_len, cpg_len,
                      nb_unit=nb_unit, compile=False)
    return model


def read_data(path, model, cache_size=None):
    file_, data = ut.read_hdf(path, cache_size)
    weights = dict()
    for k in model.output_order:
        weights[k] = get_sample_weights(data[k])
    ut.to_view(data)
    ut.to_view(weights)
    check_weights(model, data, weights)
    return (file_, data, weights)


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
            '-p', '--out_pickle',
            help='Pickle model',
            default='model.pkl',
            nargs='?')
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
            '--cpg_model',
            help='Reuse weights of cpg module',
            nargs='+')
        p.add_argument(
            '--seq_model',
            help='Reuse weights of seq module',
            nargs='+')
        p.add_argument(
            '--not_trainable',
            help='Do not train nodes with given name',
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
            '--batch_size_auto',
            help='Determine maximum batch size automatically',
            action='store_true')
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
            '--no_jump',
            help='Do not jump in training set',
            action='store_true')
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
            '--compile',
            help='Force model compilation',
            action='store_true')
        p.add_argument(
            '--eval',
            help='Evaluate performance after training',
            choices=['train', 'val'],
            default='val',
            nargs='+')
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
            (512, 0.000125),
            (256, 0.00025),
            (128, 0.0005),
            (64, 0.001),
            (32, 0.001),
            (16, 0.001)
        ]
        idx = None
        nb_sample = list(data.values())[0].shape[0]
        for i in range(len(configs)):
            batch_size = configs[i][0]
            if batch_size > nb_sample:
                continue
            self.log.info('Try batch size %d' % (batch_size))
            batch_data = {k: v[:batch_size] for k, v in data.items()}
            batch_weights = dict()
            for k, v in sample_weights.items():
                batch_weights[k] = v[:batch_size]
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

    def callbacks(self, model):
        opts = self.opts
        cbacks = []

        cbacks.append(cbk.ProgressLogger())
        cbacks.append(cbk.EarlyStopping(patience=opts.early_stop, verbose=1))
        if opts.max_time is not None:
            cbacks.append(cbk.Timer(opts.max_time * 3600 * 0.8))

        h = ModelCheckpoint(pt.join(opts.out_dir, 'model_weights_last.h5'),
                            save_best_only=False)
        cbacks.append(h)
        h = ModelCheckpoint(pt.join(opts.out_dir, 'model_weights.h5'),
                            save_best_only=True, verbose=1)
        cbacks.append(h)

        def lr_schedule():
            old_lr = model.optimizer.lr.get_value()
            new_lr = old_lr * opts.lr_decay
            model.optimizer.lr.set_value(new_lr)
            print('Learning rate dropped from %g to %g' % (old_lr, new_lr))

        h = cbk.LearningRateScheduler(lr_schedule, patience=opts.lr_schedule)
        cbacks.append(h)

        def save_lc():
            log = {'lc.csv': perf_logger.frame(),
                   'lc_batch.csv': perf_logger.batch_frame()}
            for k, v in log.items():
                with open(pt.join(opts.out_dir, k), 'w') as f:
                    f.write(perf_logs_str(v))

        perf_logger = cbk.PerformanceLogger(callbacks=[save_lc])
        cbacks.append(perf_logger)

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

        pd.set_option('display.width', 150)
        self.log = log
        self.opts = opts

        # Create output directory if not existing
        if not pt.exists(opts.out_dir):
            os.makedirs(opts.out_dir, exist_ok=True)

        # Build model
        targets = ut.read_targets(opts.train_file, opts.targets)
        if len(targets['name']) == 0:
            raise 'No targets match selection!'
        if opts.params is not None:
            model_params = Params.from_yaml(opts.params)
        else:
            model_params = None

        if opts.model is None:
            log.info('Build model from scratch')
            if model_params is None:
                assert 'Parameter file needed!'
            model = build_model(model_params, opts.train_file, targets['id'])
        else:
            log.info('Loading model')
            model = mod.model_from_list(opts.model, compile=False)

        if opts.cpg_model is not None:
            log.info('Copy cpg weights')
            cpg_model = mod.model_from_list(opts.cpg_model, compile=False)
            t = mod.copy_weights(cpg_model, model, 'c_')
            log.info('Weights copied from %d nodes' % (t))

        if opts.seq_model is not None:
            log.info('Copy seq weights')
            seq_model = mod.model_from_list(opts.seq_model, compile=False)
            t = mod.copy_weights(seq_model, model, 's_')
            log.info('Weights copied from %d nodes' % (t))

        if opts.not_trainable is not None:
            print('\nNodes excluded from training:')
            for k, v in model.nodes.items():
                if k in opts.not_trainable:
                    print(k)
                    v.trainable = False

        # Compile model
        if opts.compile or not hasattr(model, 'loss'):
            log.info('Compile model')
            if model_params is None:
                optimizer = mod.optimizer_from_json(opts.model[0])
            else:
                optimizer = mod.optimizer_from_params(model_params)
            loss = mod.loss_from_ids(model.output_order)
            model.compile(loss=loss, optimizer=optimizer)

        log.info('Save model')
        mod.model_to_json(model, pt.join(opts.out_dir, 'model.json'))
        model.save_weights(pt.join(opts.out_dir, 'model_weights.h5'),
                           overwrite=True)
        if model_params is not None:
            h = pt.join(opts.out_dir, 'configs.yaml')
            if not pt.exists(h):
                model_params.to_yaml(h)

        # Setup callbacks
        log.info('Setup callbacks')
        cbacks = self.callbacks(model)

        # Read Training data
        log.info('Read training data')
        train_file, train_data, train_weights = read_data(opts.train_file,
                                                          model, opts.max_mem)
        views = list(train_data.values()) + list(train_weights.values())
        h = cbk.DataJumper(views, nb_sample=opts.nb_sample, verbose=1,
                           jump=not opts.no_jump)
        cbacks.append(h)

        # Validation data
        log.info('Read validation data')
        if opts.val_file is None:
            val_data = train_data
            val_weights = train_weights
            val_file = None
        else:
            val_file, val_data, val_weights = read_data(opts.val_file, model,
                                                        opts.max_mem)
            views = list(val_data.values()) + list(val_weights.values())
            nb_sample = opts.nb_val_sample
            if nb_sample is None:
                nb_sample = opts.nb_sample
            h = cbk.DataJumper(views, nb_sample=nb_sample, verbose=1,
                               jump=False)
            cbacks.append(h)

        # Define batch size
        if opts.batch_size:
            batch_size = opts.batch_size
        elif opts.batch_size_auto or model_params is None:
            log.info('Adjust batch size')
            batch_size, lr = self.adjust_batch_size(model, train_data,
                                                    train_weights)
            if batch_size is None:
                log.error('GPU memory to small')
                return 1
            model.optimizer.lr.set_value(lr)
        else:
            batch_size = model_params.batch_size

        # Print infos
        print('\nInput arguments:')
        print(pu.dict_to_str(opts.__dict__))

        if model_params is not None:
            print('\nModel parameters:')
            print(model_params)

        print('\nTargets:')
        for output in model.output_order:
            h = targets['id'].index(output.replace('_y', ''))
            print('%s: %s' % (targets['id'][h], targets['name'][h]))

        print()
        print('%d training samples' % (list(train_data.values())[0].shape[0]))
        print('%d validation samples' % (list(val_data.values())[0].shape[0]))

        # Train model
        log.info('Train model')
        model.fit(data=train_data,
                  sample_weight=train_weights,
                  val_data=val_data,
                  val_sample_weight=val_weights,
                  batch_size=batch_size,
                  shuffle=opts.shuffle,
                  nb_epoch=opts.nb_epoch,
                  callbacks=cbacks,
                  verbose=0,
                  logger=lambda x: log.debug(x))

        # Use best weights on validation set
        h = pt.join(opts.out_dir, 'model_weights.h5')
        if pt.isfile(h):
            model.load_weights(h)

        if opts.out_pickle is not None:
            log.info('Pickle model')
            h = opts.out_pickle
            if pt.dirname(opts.out_pickle) == '':
                h = pt.join(opts.out_dir, h)
            mod.model_to_pickle(model, h)

        if opts.nb_epoch > 0:
            print('\n\nLearning curve:')
            for cback in cbacks:
                if isinstance(cback, cbk.PerformanceLogger):
                    h = cback
                    break
            print(perf_logs_str(h.frame()))

        if opts.eval is not None and 'train' in opts.eval:
            log.info('Evaluate training set performance')
            z = model.predict(train_data, batch_size=batch_size)
            print('\nTraining set performance:')
            eval_io(model, train_data, z, pt.join(opts.out_dir, 'train'),
                    targets)

        if opts.eval is not None and 'val' in opts.eval:
            log.info('Evaluate validation set performance')
            z = model.predict(val_data, batch_size=batch_size)
            print('\nValidation set performance:')
            eval_io(model, val_data, z, pt.join(opts.out_dir, 'val'), targets)

        train_file.close()
        if val_file:
            val_file.close()

        log.info('Done!')

        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
