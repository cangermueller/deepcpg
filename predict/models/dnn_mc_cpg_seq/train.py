#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import keras.models as kmodels
from keras.layers import core as kcore
from keras.layers import convolutional as kconv
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.optimizers as kopt
import pandas as pd
import numpy as np
import yaml

from predict.evaluation import eval_to_str
from utils import evaluate_all, read_data, load_model, MASK
from callbacks import LearningRateScheduler, PerformanceLogger


class CpgNetParams(object):

    def __init__(self):
        self.dim = [200, 1]
        self.num_filters = 4
        self.filter_len = 4
        self.pool_len = 2
        self.num_hidden = 32
        self.dropout = 0.2

    def update(self, params):
        self.__dict__.update(params)

    def __str__(self):
        return str(vars(self))


class SeqNetParams(object):

    def __init__(self):
        self.dim = [1001]
        self.num_filters = 4
        self.filter_len = 8
        self.pool_len = 4
        self.num_hidden = 32
        self.dropout = 0.2

    def update(self, params):
        self.__dict__.update(params)

    def __str__(self):
        return str(vars(self))


class NetParams(object):

    def __init__(self):
        self.seq = SeqNetParams()
        self.cpg = CpgNetParams()

        self.num_hidden = 16
        self.num_targets = 1
        self.dropout = 0.25

        self.lr = 0.01
        self.lr_decay = 0.5
        self.early_stop = 3
        self.max_epochs = 10

    def update(self, params):
        params = dict(params)
        if 'seq' in params:
            self.seq.update(params['seq'])
            del params['seq']
        if 'cpg' in params:
            self.cpg.update(params['cpg'])
            del params['cpg']
        self.__dict__.update(params)
        self.set_targets(self.num_targets)

    def set_targets(self, num_targets):
        self.cpg.dim[1] = self.num_targets
        self.num_targets = num_targets

    def __str__(self):
        return str(vars(self))


def label(prefix, x):
    return '%s_%s' % (prefix, x)


def add_seq_conv(model, params, prefix='s'):
    def lab(x):
        return '%s_%s' % (prefix, x)

    model.add_input(name=lab('x'), ndim=3)
    # conv
    layer = kconv.Convolution1D(4, params.num_filters,
                            params.filter_len,
                            activation='relu',
                            init='glorot_uniform',
                            border_mode='same')
    model.add_node(input=lab('x'), name=lab('c1'), layer=layer)
    # pool
    layer = kconv.MaxPooling1D(pool_length=params.pool_len)
    model.add_node(input=lab('c1'), name=lab('p1'), layer=layer)
    # flatten
    model.add_node(input=lab('p1'), name=lab('f1'), layer=kcore.Flatten())
    model.add_node(
    input=lab('f1'),
    name=lab('f1d'),
    layer=kcore.Dropout(
        params.dropout))
    # hidden
    nconv = (params.dim[0] // params.pool_len) * params.num_filters
    layer = kcore.Dense(nconv, params.num_hidden,
                    activation='relu',
                    init='glorot_uniform')
    model.add_node(input=lab('f1d'), name=lab('h1'), layer=layer)
    model.add_node(
    input=lab('h1'),
    name=lab('h1d'),
    layer=kcore.Dropout(
        params.dropout))


def add_cpg_conv(model, params, prefix='c'):
    def lab(x):
        return '%s_%s' % (prefix, x)

    model.add_input(name=lab('x'), ndim=4)
    # conv
    layer = kconv.Convolution2D(params.num_filters, 2,
                            params.filter_len, 1,
                            activation='relu',
                            init='glorot_uniform',
                            border_mode='same')
    model.add_node(input=lab('x'), name=lab('c1'), layer=layer)
    # pool
    layer = kconv.MaxPooling2D(poolsize=(1, params.pool_len))
    model.add_node(input=lab('c1'), name=lab('p1'), layer=layer)
    # flatten
    model.add_node(input=lab('p1'), name=lab('f1'), layer=kcore.Flatten())
    model.add_node(
    input=lab('f1'),
    name=lab('f1d'),
    layer=kcore.Dropout(
        params.dropout))
    # hidden
    nconv = params.dim[1] * (params.dim[0] //
                             params.pool_len) * params.num_filters
    layer = kcore.Dense(nconv, params.num_hidden,
                    activation='relu',
                    init='glorot_uniform')
    model.add_node(input=lab('f1d'), name=lab('h1'), layer=layer)
    model.add_node(
    input=lab('h1'),
    name=lab('h1d'),
    layer=kcore.Dropout(
        params.dropout))


def add_out(model, params, prefix):
    def lab(x):
        return label(prefix, x)

    prev_names = [label('s', 'h1d'), label('c', 'h1d')]
    prev_dim = params.seq.num_hidden + params.cpg.num_hidden
    if params.num_hidden:
        layer = kcore.Dense(prev_dim, params.num_hidden,
                            activation='relu',
                            init='glorot_uniform')
        model.add_node(inputs=prev_names,
                       name=lab('h1'),
                       layer=layer)
        layer = kcore.Dropout(params.dropout)
        model.add_node(input=lab('h1'), name=lab('h1d'), layer=layer)
        prev_names = [lab('h1d')]
        prev_dim = params.num_hidden

    layer = kcore.Dense(prev_dim, 1,
                        activation='sigmoid',
                        init='glorot_uniform')
    if len(prev_names) == 1:
        model.add_node(input=prev_names[0], name=lab('z'), layer=layer)
    else:
        model.add_node(inputs=prev_names, name=lab('z'), layer=layer)
    model.add_output(input=lab('z'), name=lab('y'))


def build_model(params):
    model = kmodels.Graph()
    add_seq_conv(model, params.seq, 's')
    add_cpg_conv(model, params.cpg, 'c')
    units = []
    for i in range(params.num_targets):
        u = 'u%d' % (i)
        units.append(u)
        add_out(model, params, u)
    loss = {label(u, 'y'): 'binary_crossentropy' for u in units}
    opt = kopt.Adam(params.lr)
    model.compile(loss=loss, optimizer=opt)
    return model


def sample_weights(y, outputs):
    w = dict()
    for o in outputs:
        yo = y[o]
        wo = np.zeros(len(yo), dtype='float32')
        wo[yo != MASK] = 1
        w[o] = wo
    return w


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
            'train_data',
            help='Training data file')
        p.add_argument(
            '--val_data',
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
            '--log_train',
            help='Log training set performance each epoch',
            action='store_true')
        p.add_argument(
            '--log_val',
            help='Log validation set performance each epoch',
            action='store_true')
        p.add_argument(
            '--max_samples',
            help='Limit # samples',
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
        log=logging.getLogger(name)
        if opts.verbose:
            log.setLevel(logging.DEBUG)
        else:
            log.setLevel(logging.INFO)
            log.debug(opts)


        if opts.seed is not None:
            np.random.seed(opts.seed)
        pd.set_option('display.width', 150)

        log.info('Read data')
        train_data=read_data(opts.train_data, opts.max_samples)
        num_targets=len([x for x in train_data.keys() if x.startswith('u')])
        print('%d training samples' % (len(train_data['pos'])))

        if opts.val_data is None:
            val_data=train_data
        else:
            val_data=read_data(opts.val_data)
            print('%d validation samples' % (len(val_data['pos'])))

        model_params=NetParams()
        if opts.model_params:
            with open(opts.model_params, 'r') as f:
                configs=yaml.load(f.read())
                configs['num_targets']=num_targets
                model_params.update(configs)

        print('Model parameters:')
        print(model_params)
        print()

        if opts.model_file:
            log.info('Build model from file')
            model=load_model(opts.model_file)
        else:
            log.info('Build model')
            model=build_model(model_params)
            with open(pt.join(opts.out_dir, 'model.json'), 'w') as f:
                f.write(model.to_json())

        sample_weight_train=sample_weights(train_data, model.output_order)
        sample_weight_val=sample_weights(val_data, model.output_order)

        log.info('Fit model')
        model_weights_last=pt.join(opts.out_dir, 'model_weights_last.h5')
        model_weights_best=pt.join(opts.out_dir, 'model_weights_best.h5')

        cb=[]
        cb.append(ModelCheckpoint(model_weights_last, save_best_only=False))
        cb.append(ModelCheckpoint(model_weights_best,
                                  save_best_only=True, verbose=1))
        cb.append(EarlyStopping(patience=model_params.early_stop, verbose=1))

        def lr_schedule():
            old_lr=opt.lr.get_value()
            new_lr=old_lr * model_params.lr_decay
            opt.lr.set_value(new_lr)
            print('Learning rate dropped from %.4f to %.4f' % (old_lr, new_lr))

        cb.append(LearningRateScheduler(lr_schedule,
                                        patience=model_params.early_stop - 1))

        if opts.log_train:
            cb.append(PerformanceLogger(train_data, label='train'))
        if opts.log_val:
            cb.append(PerformanceLogger(val_data, label='val'))

        model.fit(train_data, validation_data = val_data,
                  callbacks = cb,
                  nb_epoch = model_params.max_epochs,
                  verbose = 2,
                  sample_weight = sample_weight_train,
                  sample_weight_val = sample_weight_val)

        if pt.isfile(model_weights_best):
            model.load_weights(model_weights_best)

        print('\nTraining set performance:')
        z_train=model.predict(train_data)
        p_train=evaluate_all(train_data, z_train)
        t=eval_to_str(p_train)
        print(t)
        with open(pt.join(opts.out_dir, 'perf_train.csv'), 'w') as f:
            f.write(t)

        print('\nValidation set loss:')
        z_val=model.predict(val_data)
        p_val=evaluate_all(val_data, z_val)
        t=eval_to_str(p_val)
        print(t)
        with open(pt.join(opts.out_dir, 'perf_val.csv'), 'w') as f:
            f.write(t)

        log.info('Done!')

        return 0


if __name__ == '__main__':
    app=App()
    app.run(sys.argv)
