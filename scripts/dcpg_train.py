#!/usr/bin/env python

"""Train a DeepCpG model to predict DNA methylation.

Trains a DeepCpG model on DNA (DNA model), neighboring methylation states
(CpG model), or both (Joint model) to predict CpG methylation of multiple cells.
Allows to fine-tune individual models or to train them from scratch.

Examples
--------
Train a DNA model on chromosome 1, 3, and 5,  and use chromosome 13, 14, and
15 for validation:

.. code:: bash

    dcpg_train.py
        ./data/c{1,3,5}_*.h5
        --val_files ./data/c{13,14,15}_*.h5
        --dna_model CnnL2h128
        --out_dir ./models/dna

Train a CpG model:

.. code:: bash

    dcpg_train.py
        ./data/c{1,3,5}_*.h5
        --val_files ./data/c{13,14,15}_*.h5
        --cpg_model RnnL1
        --out_dir ./models/cpg

Train a Joint model using a pre-trained DNA and CpG model:

.. code:: bash

    dcpg_train.py
        ./data/c{1,3,5}_*.h5
        --val_files ./data/c{13,14,15}_*.h5
        --dna_model ./models/dna
        --cpg_model ./models/cpg
        --joint_model JointL2h512
        --train_models joint
        --out_dir ./models/joint

See Also
--------
* ``dcpg_eval.py``: For evaluating a trained model and imputing methylation
    profiles.
"""

from __future__ import print_function
from __future__ import division

from collections import OrderedDict
import os
import random
import re
import sys

import argparse
import h5py as h5
import logging
import numpy as np
import pandas as pd
import six
from six.moves import range

from keras import backend as K
from keras import callbacks as kcbk
from keras.models import Model
from keras.optimizers import Adam

from deepcpg import callbacks as cbk
from deepcpg import data as dat
from deepcpg import metrics as met
from deepcpg import models as mod
from deepcpg.data import hdf, OUTPUT_SEP
from deepcpg.utils import format_table, make_dir, EPS


LOG_PRECISION = 4

CLA_METRICS = [met.acc]

REG_METRICS = [met.mse, met.mae]


def remove_outputs(model):
    while model.layers[-1] in model.output_layers:
        model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []
    model.output_names = None


def rename_layers(model, scope=None):
    if not scope:
        scope = model.scope
    for layer in model.layers:
        if layer in model.input_layers or layer.name.startswith(scope):
            continue
        layer.name = '%s/%s' % (scope, layer.name)


def get_output_stats(output):
    stats = OrderedDict()
    output = np.ma.masked_values(output, dat.CPG_NAN)
    stats['nb_tot'] = len(output)
    stats['nb_obs'] = np.sum(output != dat.CPG_NAN)
    stats['frac_obs'] = stats['nb_obs'] / stats['nb_tot']
    stats['mean'] = float(np.mean(output))
    stats['var'] = float(np.var(output))
    return stats


def get_output_weights(output_names, weight_patterns):
    regex_weights = dict()
    for weight_pattern in weight_patterns:
        tmp = [tmp.strip() for tmp in weight_pattern.split('=')]
        if len(tmp) != 2:
            raise ValueError('Invalid weight pattern "%s"!' % (weight_pattern))
        regex_weights[tmp[0]] = float(tmp[1])

    output_weights = dict()
    for output_name in output_names:
        for regex, weight in six.iteritems(regex_weights):
            if re.match(regex, output_name):
                output_weights[output_name] = weight
        if output_name not in output_weights:
            output_weights[output_name] = 1.0
    return output_weights


def get_class_weights(labels, nb_class=None):
    freq = np.bincount(labels) / len(labels)

    if nb_class is None:
        nb_class = len(freq)

    if len(freq) < nb_class:
        tmp = np.zeros(nb_class, dtype=freq.dtype)
        tmp[:len(freq)] = freq
        freq = tmp
    weights = 1 / (freq + EPS)
    weights /= weights.sum()
    return weights


def get_output_class_weights(output_name, output):
    output = output[output != dat.CPG_NAN]
    _output_name = output_name.split(OUTPUT_SEP)
    if _output_name[0] == 'cpg':
        weights = get_class_weights(output, 2)
    elif _output_name[-1] == 'cat_var':
        weights = get_class_weights(output, 3)
    elif _output_name[-1] in ['cat2_var', 'diff', 'mode']:
        weights = get_class_weights(output, 2)
    else:
        return None
    weights = OrderedDict(zip(range(len(weights)), weights))
    return weights


def perf_logs_str(logs):
    t = logs.to_csv(None, sep='\t', float_format='%.4f', index=False)
    return t


def get_metrics(output_name):
    _output_name = output_name.split(OUTPUT_SEP)
    if _output_name[0] == 'cpg':
        metrics = CLA_METRICS
    elif _output_name[0] == 'bulk':
        metrics = REG_METRICS + CLA_METRICS
    elif _output_name[-1] in ['diff', 'mode', 'cat2_var']:
        metrics = CLA_METRICS
    elif _output_name[-1] == 'mean':
        metrics = REG_METRICS + CLA_METRICS
    elif _output_name[-1] == 'var':
        metrics = REG_METRICS
    elif _output_name[-1] == 'cat_var':
        metrics = [met.cat_acc]
    else:
        raise ValueError('Invalid output name "%s"!' % output_name)
    return metrics


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
            description='Trains model on DNA (DNA model), neighboring '
            'methylation states (CpG model), or both (Joint model) to predict '
            'CpG methylation of multiple cells.')

        # IO
        g = p.add_argument_group('input-output arguments')
        g.add_argument(
            'train_files',
            nargs='+',
            help='Training data files')
        g.add_argument(
            '--val_files',
            nargs='+',
            help='Validation data files')
        g.add_argument(
            '-o', '--out_dir',
            default='./train',
            help='Output directory')

        g = p.add_argument_group('arguments to define the model architecture')
        models = sorted(list(mod.dna.list_models().keys()))
        g.add_argument(
            '--dna_model',
            help='Name of DNA model or files of existing model.'
            ' Available models: %s' % ', '.join(models),
            nargs='+')
        g.add_argument(
            '--dna_wlen',
            help='DNA window length',
            type=int)
        models = sorted(list(mod.cpg.list_models().keys()))
        g.add_argument(
            '--cpg_model',
            help='Name of CpG model or files of existing model.'
            ' Available models: %s' % ', '.join(models),
            nargs='+')
        g.add_argument(
            '--cpg_wlen',
            help='CpG window length',
            type=int)
        models = sorted(list(mod.joint.list_models().keys()))
        g.add_argument(
            '--joint_model',
            help='Name of Joint model.'
            ' Available models: %s' % ', '.join(models),
            default='JointL2h512')
        g.add_argument(
            '--model_files',
            help='Files of existing model',
            nargs='+')

        g = p.add_argument_group('arguments to define which model components '
                                 'are trained')
        g.add_argument(
            '--fine_tune',
            help='Only train output layers',
            action='store_true')
        g.add_argument(
            '--train_models',
            help='Only train the specified models',
            choices=['dna', 'cpg', 'joint'],
            nargs='+')
        g.add_argument(
            '--trainable',
            help='Regex of layers that should be trained',
            nargs='+')
        g.add_argument(
            '--not_trainable',
            help='Regex of layers that should not be trained',
            nargs='+')
        g.add_argument(
            '--freeze_filter',
            help='Exclude filter weights of first convolutional layer from '
            'training',
            action='store_true')
        g.add_argument(
            '--filter_weights',
            help='HDF5 file with weights to be used for initializing filters',
            nargs='+')

        g = p.add_argument_group('training arguments')
        g.add_argument(
            '--learning_rate',
            help='Learning rate',
            type=float,
            default=0.0001)
        g.add_argument(
            '--learning_rate_decay',
            help='Exponential learning rate decay factor',
            type=float,
            default=0.975)
        g.add_argument(
            '--nb_epoch',
            help='Maximum # training epochs',
            type=int,
            default=30)
        g.add_argument(
            '--nb_train_sample',
            help='Maximum # training samples',
            type=int)
        g.add_argument(
            '--nb_val_sample',
            help='Maximum # validation samples',
            type=int)
        g.add_argument(
            '--batch_size',
            help='Batch size',
            type=int,
            default=128)
        g.add_argument(
            '--early_stopping',
            help='Early stopping patience',
            type=int,
            default=5)
        g.add_argument(
            '--dropout',
            help='Dropout rate',
            type=float,
            default=0.0)
        g.add_argument(
            '--l1_decay',
            help='L1 weight decay',
            type=float,
            default=0.0001)
        g.add_argument(
            '--l2_decay',
            help='L2 weight decay',
            type=float,
            default=0.0001)
        g.add_argument(
            '--no_tensorboard',
            help='Do not store Tensorboard summaries',
            action='store_true')

        g = p.add_argument_group('arguments to select outputs and weights')
        g.add_argument(
            '--output_names',
            help='Regex to select outputs',
            nargs='+',
            default=['cpg/.*'])
        g.add_argument(
            '--nb_output',
            type=int,
            help='Maximum number of outputs')
        g.add_argument(
            '--no_class_weights',
            help='Do not weight classes',
            action='store_true')
        g.add_argument(
            '--output_weights',
            help='Output weights defined as a list of `output`=`weight` '
            'patterns, where `output` is a regex of output names, and '
            '`weight` the weight that is assigned to them',
            nargs='+')
        g.add_argument(
            '--replicate_names',
            help='Regex to select replicates',
            nargs='+')
        g.add_argument(
            '--nb_replicate',
            type=int,
            help='Maximum number of replicates')

        g = p.add_argument_group('advanced arguments')
        g.add_argument(
            '--max_time',
            help='Maximum training time in hours',
            type=float)
        g.add_argument(
            '--stop_file',
            help='File that terminates training if it exists')
        g.add_argument(
            '--seed',
            help='Seed of random number generator',
            type=int,
            default=0)
        g.add_argument(
            '--no_log_outputs',
            help='Do not log performance metrics of individual outputs',
            action='store_true')
        g.add_argument(
            '--verbose',
            help='More detailed log messages',
            action='store_true')
        g.add_argument(
            '--log_file',
            help='Write log messages to file')
        g.add_argument(
            '--data_q_size',
            help='Size of data generator queue',
            type=int,
            default=10)
        g.add_argument(
            '--data_nb_worker',
            help='Number of worker for data generator queue',
            type=int,
            default=1)
        return p

    def get_callbacks(self):
        opts = self.opts
        callbacks = []

        if opts.val_files:
            callbacks.append(kcbk.EarlyStopping(
                'val_loss' if opts.val_files else 'loss',
                patience=opts.early_stopping,
                verbose=1
            ))

        callbacks.append(kcbk.ModelCheckpoint(
            os.path.join(opts.out_dir, 'model_weights_train.h5'),
            save_best_only=False))
        monitor = 'val_loss' if opts.val_files else 'loss'
        callbacks.append(kcbk.ModelCheckpoint(
            os.path.join(opts.out_dir, 'model_weights_val.h5'),
            monitor=monitor,
            save_best_only=True, verbose=1
        ))

        max_time = int(opts.max_time * 3600) if opts.max_time else None
        callbacks.append(cbk.TrainingStopper(
            max_time=max_time,
            stop_file=opts.stop_file,
            verbose=1
        ))

        def learning_rate_schedule(epoch):
            lr = opts.learning_rate * opts.learning_rate_decay**epoch
            print('Learning rate: %.3g' % lr)
            return lr

        callbacks.append(kcbk.LearningRateScheduler(learning_rate_schedule))

        def save_lc(epoch, epoch_logs, val_epoch_logs):
            logs = {'lc_train.tsv': epoch_logs,
                    'lc_val.tsv': val_epoch_logs}
            for name, logs in six.iteritems(logs):
                if not logs:
                    continue
                logs = pd.DataFrame(logs)
                with open(os.path.join(opts.out_dir, name), 'w') as f:
                    f.write(perf_logs_str(logs))

        metrics = OrderedDict()
        for metric_funs in six.itervalues(self.metrics):
            for metric_fun in metric_funs:
                metrics[metric_fun.__name__] = True
        metrics = ['loss'] + list(metrics.keys())

        self.perf_logger = cbk.PerformanceLogger(
            callbacks=[save_lc],
            metrics=metrics,
            precision=LOG_PRECISION,
            verbose=not opts.no_log_outputs
        )
        callbacks.append(self.perf_logger)

        if K._BACKEND == 'tensorflow' and not opts.no_tensorboard:
            callbacks.append(kcbk.TensorBoard(
                log_dir=opts.out_dir,
                histogram_freq=0,
                write_graph=True,
                write_images=True
            ))

        return callbacks

    def print_output_stats(self, output_stats):
        table = OrderedDict()
        for name, stats in six.iteritems(output_stats):
            table.setdefault('name', []).append(name)
            for key in stats:
                table.setdefault(key, []).append(stats[key])
        print('Output statistics:')
        print(format_table(table))
        print()

    def print_class_weights(self, class_weights):
        table = OrderedDict()
        for name, class_weight in six.iteritems(class_weights):
            if not class_weight:
                continue
            column = []
            for cla, weight in six.iteritems(class_weight):
                column.append('%s=%.2f' % (cla, weight))
                table[name] = column
        if table:
            print('Class weights:')
            print(format_table(table))
            print()

    def build_dna_model(self):
        opts = self.opts
        log = self.log
        if os.path.exists(opts.dna_model[0]):
            log.info('Loading existing DNA model ...')
            dna_model = mod.load_model(opts.dna_model, log=log.info)
            remove_outputs(dna_model)
            rename_layers(dna_model, 'dna')
        else:
            log.info('Building DNA model ...')
            dna_model_builder = mod.dna.get(opts.dna_model[0])(
                l1_decay=opts.l1_decay,
                l2_decay=opts.l2_decay,
                dropout=opts.dropout)
            dna_wlen = dat.get_dna_wlen(opts.train_files[0], opts.dna_wlen)
            dna_inputs = dna_model_builder.inputs(dna_wlen)
            dna_model = dna_model_builder(dna_inputs)
        return dna_model

    def build_cpg_model(self):
        opts = self.opts
        log = self.log

        replicate_names = dat.get_replicate_names(
            opts.train_files[0],
            regex=opts.replicate_names,
            nb_key=opts.nb_replicate)
        if not replicate_names:
            raise ValueError('No replicates found!')
        print('Replicate names:')
        print(', '.join(replicate_names))
        print()

        cpg_wlen = dat.get_cpg_wlen(opts.train_files[0], opts.cpg_wlen)

        if os.path.exists(opts.cpg_model[0]):
            log.info('Loading existing CpG model ...')
            src_cpg_model = mod.load_model(opts.cpg_model, log=log.info)
            remove_outputs(src_cpg_model)
            rename_layers(src_cpg_model, 'cpg')
            nb_replicate = src_cpg_model.input_shape[0][1]
            if nb_replicate != len(replicate_names):
                tmp = 'CpG model was trained with %d replicates but %d'
                'replicates provided. Copying weight to new model ...'
                tmp %= (nb_replicate, len(replicate_names))
                log.info('Replicate names differ: '
                         'Copying weights to new model ...')
                cpg_model_builder = mod.cpg.get(src_cpg_model.name)(
                    l1_decay=opts.l1_decay,
                    l2_decay=opts.l2_decay,
                    dropout=opts.dropout)
                cpg_inputs = cpg_model_builder.inputs(cpg_wlen, replicate_names)
                cpg_model = cpg_model_builder(cpg_inputs)
                mod.copy_weights(src_cpg_model, cpg_model)
            else:
                cpg_model = src_cpg_model
        else:
            log.info('Building CpG model ...')
            cpg_model_builder = mod.cpg.get(opts.cpg_model[0])(
                l1_decay=opts.l1_decay,
                l2_decay=opts.l2_decay,
                dropout=opts.dropout)
            cpg_inputs = cpg_model_builder.inputs(cpg_wlen, replicate_names)
            cpg_model = cpg_model_builder(cpg_inputs)

        return cpg_model

    def build_model(self):
        opts = self.opts
        log = self.log

        output_names = dat.get_output_names(opts.train_files[0],
                                            regex=opts.output_names,
                                            nb_key=opts.nb_output)
        if not output_names:
            raise ValueError('No outputs found!')

        dna_model = None
        if opts.dna_model:
            dna_model = self.build_dna_model()

        cpg_model = None
        if opts.cpg_model:
            cpg_model = self.build_cpg_model()

        if dna_model is not None and cpg_model is not None:
            log.info('Joining models ...')
            joint_model_builder = mod.joint.get(opts.joint_model)(
                l1_decay=opts.l1_decay,
                l2_decay=opts.l2_decay,
                dropout=opts.dropout)
            stem = joint_model_builder([dna_model, cpg_model])
            stem.name = '_'.join([stem.name, dna_model.name, cpg_model.name])
        elif dna_model is not None:
            stem = dna_model
        elif cpg_model is not None:
            stem = cpg_model
        else:
            log.info('Loading existing model ...')
            stem = mod.load_model(opts.model_files, log=log.info)
            if sorted(output_names) == sorted(stem.output_names):
                return stem
            log.info('Removing existing output layers ...')
            remove_outputs(stem)

        outputs = mod.add_output_layers(stem.outputs[0], output_names)
        model = Model(stem.inputs, outputs, stem.name)
        return model

    def set_trainability(self, model):
        opts = self.opts
        trainable = []
        not_trainable = []
        if opts.fine_tune:
            not_trainable.append('.*')
        elif opts.train_models:
            not_trainable.append('.*')
            for name in opts.train_models:
                trainable.append('%s/' % name)
        if opts.freeze_filter:
            not_trainable.append(mod.get_first_conv_layer(model.layers).name)
        if not trainable and opts.trainable:
            trainable = opts.trainable
        if not not_trainable and opts.not_trainable:
            not_trainable = opts.not_trainable

        if not trainable and not not_trainable:
            return

        table = OrderedDict()
        table['layer'] = []
        table['trainable'] = []
        for layer in model.layers:
            if layer not in model.input_layers + model.output_layers:
                if not hasattr(layer, 'trainable'):
                    continue
                for regex in not_trainable:
                    if re.match(regex, layer.name):
                        layer.trainable = False
                for regex in trainable:
                    if re.match(regex, layer.name):
                        layer.trainable = True
                table['layer'].append(layer.name)
                table['trainable'].append(layer.trainable)
        print('Layer trainability:')
        print(format_table(table))
        print()

    def init_filter_weights(self, filename, conv_layer):
        h5_file = h5.File(filename[0], 'r')
        group = h5_file
        if len(filename) > 1:
            group = h5_file[filename[1]]
        weights = group['weights'].value
        bias = None
        if 'bias' in group:
            bias = group['bias'].value
        h5_file.close()

        assert weights.ndim == 4
        if weights.shape[1] != 1:
            weights = weights[:, :, :, 0]
            weights = np.swapaxes(weights, 0, 2)
            weights = np.expand_dims(weights, 1)

        # filter_size x 1 x 4 x nb_filter
        cur_weights, cur_bias = conv_layer.get_weights()

        # Adapt number of filters
        tmp = min(weights.shape[-1], cur_weights.shape[-1])
        weights = weights[:, :, :, :tmp]

        # Adapt filter size
        if len(weights) > len(cur_weights):
            # Truncate weights
            idx = (len(weights) - len(cur_weights)) // 2
            weights = weights[idx:(idx + len(cur_weights))]
        elif len(weights) < len(cur_weights):
            # Pad weights
            shape = [len(cur_weights)] + list(weights.shape[1:])
            pad_weights = np.random.uniform(0, 1, shape) * 1e-2
            idx = (len(cur_weights) - len(weights)) // 2
            pad_weights[idx:(idx + len(weights))] = weights
            weights = pad_weights

        assert np.all(weights.shape[:-1] == cur_weights.shape[:-1])
        cur_weights[:, :, :, :weights.shape[-1]] = weights
        if bias is not None:
            bias = bias[:len(cur_bias)]
            cur_bias[:len(bias)] = bias

        conv_layer.set_weights((cur_weights, cur_bias))
        print('%d filters initialized' % weights.shape[-1])

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

        make_dir(opts.out_dir)

        log.info('Building model ...')
        model = self.build_model()

        model.summary()
        self.set_trainability(model)
        if opts.filter_weights:
            conv_layer = mod.get_first_conv_layer(model.layers)
            log.info('Initializing filters of %s ...' % conv_layer.name)
            self.init_filter_weights(opts.filter_weights, conv_layer)
        mod.save_model(model, os.path.join(opts.out_dir, 'model.json'))

        log.info('Computing output statistics ...')
        output_names = []
        for output_layer in model.output_layers:
            output_names.append(output_layer.name)

        output_stats = OrderedDict()

        if opts.no_class_weights:
            class_weights = None
        else:
            class_weights = OrderedDict()

        for name in output_names:
            output = hdf.read(opts.train_files, 'outputs/%s' % name,
                              nb_sample=opts.nb_train_sample)
            output = list(output.values())[0]
            output_stats[name] = get_output_stats(output)
            if class_weights is not None:
                class_weights[name] = get_output_class_weights(name, output)

        self.print_output_stats(output_stats)
        if class_weights:
            self.print_class_weights(class_weights)

        output_weights = None
        if opts.output_weights:
            log.info('Initializing output weights ...')
            output_weights = get_output_weights(output_names,
                                                opts.output_weights)
            print('Output weights:')
            for output_name in output_names:
                if output_name in output_weights:
                    print('%s: %.2f' % (output_name,
                                        output_weights[output_name]))
            print()

        self.metrics = dict()
        for output_name in output_names:
            self.metrics[output_name] = get_metrics(output_name)

        optimizer = Adam(lr=opts.learning_rate)
        model.compile(optimizer=optimizer,
                      loss=mod.get_objectives(output_names),
                      loss_weights=output_weights,
                      metrics=self.metrics)

        log.info('Loading data ...')
        replicate_names = dat.get_replicate_names(
            opts.train_files[0],
            regex=opts.replicate_names,
            nb_key=opts.nb_replicate)
        data_reader = mod.data_reader_from_model(
            model, replicate_names=replicate_names)
        nb_train_sample = dat.get_nb_sample(opts.train_files,
                                            opts.nb_train_sample)
        train_data = data_reader(opts.train_files,
                                 class_weights=class_weights,
                                 batch_size=opts.batch_size,
                                 nb_sample=nb_train_sample,
                                 shuffle=True,
                                 loop=True)

        if opts.val_files:
            nb_val_sample = dat.get_nb_sample(opts.val_files,
                                              opts.nb_val_sample)
            val_data = data_reader(opts.val_files,
                                   batch_size=opts.batch_size,
                                   nb_sample=nb_val_sample,
                                   shuffle=False,
                                   loop=True)
        else:
            val_data = None
            nb_val_sample = None

        log.info('Initializing callbacks ...')
        callbacks = self.get_callbacks()

        log.info('Training model ...')
        print()
        print('Training samples: %d' % nb_train_sample)
        if nb_val_sample:
            print('Validation samples: %d' % nb_val_sample)
        model.fit_generator(
            train_data,
            steps_per_epoch=nb_train_sample // opts.batch_size,
            epochs=opts.nb_epoch,
            callbacks=callbacks,
            validation_data=val_data,
            validation_steps=nb_val_sample // opts.batch_size,
            max_queue_size=opts.data_q_size,
            workers=opts.data_nb_worker,
            verbose=0)

        print('\nTraining set performance:')
        print(format_table(self.perf_logger.epoch_logs,
                           precision=LOG_PRECISION))

        if self.perf_logger.val_epoch_logs:
            print('\nValidation set performance:')
            print(format_table(self.perf_logger.val_epoch_logs,
                               precision=LOG_PRECISION))

        # Restore model with highest validation performance
        filename = os.path.join(opts.out_dir, 'model_weights_val.h5')
        if os.path.isfile(filename):
            model.load_weights(filename)

        # Delete metrics since they cause problems when loading the model
        # from HDF5 file. Metrics can be loaded from json + weights file.
        model.metrics = None
        model.metrics_names = None
        model.metrics_tensors = None
        model.save(os.path.join(opts.out_dir, 'model.h5'))

        log.info('Done!')

        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
