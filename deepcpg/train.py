#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt

import h5py as h5
import numpy as np

from deepcpg.data.preprocess import CPG_NAN

from keras import callbacks as kcbk



# TODO:
# Class weights
# Normalize inputs

def perf_logs_str(logs):
    t = logs.to_csv(None, sep='\t', float_format='%.4f', index=False)
    return t


def get_sample_weights(y, weight_classes=False, mask_value=CPG_NAN):
    y = y[:]
    class_weights = {mask_value: 0}
    if weight_classes:
        t = y[y != mask_value].mean()
        class_weights[0] = t
        class_weights[1] = 1 - t
    sample_weights = np.ones(y.shape, dtype='float16')
    for k, v in class_weights.items():
        sample_weights[y == k] = v
    return sample_weights


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
            '--val_file',
            nargs='+',
            help='Validation data files')
        p.add_argument(
            '-o', '--out_dir',
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
            type=float)
        p.add_argument(
            '--lr_schedule',
            help='Learning rate scheduler patience',
            type=int,
            default=1)
        p.add_argument(
            '--lr_decay',
            help='Exponential learning rate decay factor',
            type=float,
            default=0.95)
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

    def callbacks(self, model):
        opts = self.opts
        cbacks = []

        cbacks.append(kcbk.EarlyStopping(patience=opts.early_stop, verbose=1))
        if opts.max_time is not None:
            cbacks.append(cbk.Timer(opts.max_time * 3600 * 0.8))

        h = kcbk.ModelCheckpoint(pt.join(opts.out_dir, 'model_weights_last.h5'),
                                 save_best_only=False)
        cbacks.append(h)
        h = kcbk.ModelCheckpoint(pt.join(opts.out_dir, 'model_weights.h5'),
                                 save_best_only=True, verbose=1)
        cbacks.append(h)

        h = kcbk.LearningRateScheduler(
            lambda epoch: opts.lr * opts.lr_decay**epoch)
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

        # Setup callbacks
        log.info('Setup callbacks')
        cbacks = self.callbacks(model)

        train_data = data_generator(opts.train_files,
                                    batch_size=opts.batch_size,
                                    nb_sample=opts.nb_sample,
                                    targets=opts.targets)
        val_data = data_generator(opts.val_files,
                                  batch_size=opts.batch_size,
                                  nb_sample=opts.nb_val_sample,
                                  targets=opts.targets)

        model = models.BuildModel()

        # Train model
        log.info('Train model')
        fit_generator(self, generator, samples_per_epoch, nb_epoch, verbose=1, callbacks=[], validation_data=None, nb_val_samples=None, class_weight=None, max_q_size=10, nb_worker=1, pickle_safe=False)
        model.fit_generator(train_data,

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
            for cback in cbacks:
                if isinstance(cback, cbk.PerformanceLogger):
                    perf_logger = cback
                    break
            lc = perf_logger.frame()
            print('\n\nLearning curve:')
            print(perf_logs_str(lc))
            if len(lc) > 5:
                lc = lc.loc[lc.epoch > 2]
            lc.set_index('epoch', inplace=True)
            ax = lc.plot(figsize=(10, 6))
            ax.get_figure().savefig(pt.join(opts.out_dir, 'lc.png'))

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


