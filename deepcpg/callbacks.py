from collections import OrderedDict
from time import time
import warnings

from keras.callbacks import Callback
from keras import backend as kback

import pandas as pd
import numpy as np

from .utils import format_table


class EarlyStopping(Callback):
    def __init__(self, monitor='val_loss', patience=0, verbose=0):
        super(Callback, self).__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.best_score = np.inf
        self.counter = 0

    def on_epoch_end(self, epoch, logs={}):
        score = logs.get(self.monitor)
        if score is None:
            warnings.warn("Early stopping requires %s!" % (self.monitor),
                          RuntimeWarning)

        if np.isnan(score):
            if self.verbose > 0:
                print("Epoch %d: stop due to nan" % (epoch))
            self.model.stop_training = True
        elif score < self.best_score:
            self.counter = 0
            self.best_score = score
        else:
            self.counter += 1
            if self.counter > self.patience:
                if self.verbose > 0:
                    print("Epoch %d: early stopping" % (epoch))
                self.model.stop_training = True


class PerformanceLogger(Callback):

    def __init__(self, metrics=['loss', 'acc'], log_freq=0.1, callbacks=[], logger=print):
        self.metrics = metrics
        self.log_freq = log_freq
        self.callbacks = callbacks
        self.logger = logger
        self._line = '=' * 100
        self.epoch_logs = None
        self.val_epoch_logs = None
        self.batch_logs = []

    def _log(self, x):
        if self.logger:
            self.logger(x)

    def _init_logs(self, logs, train=True):
        logs = list(logs)
        if train:
            logs = [log for log in logs if not log.startswith('val_')]
        else:
            logs = [log[4:] for log in logs if log.startswith('val_')]
        avg = []
        outputs = []
        for metric in self.metrics:
            if metric in logs:
                avg.append(metric)
            else:
                outputs.extend(sorted([log for log in logs if log.endswith('_' + metric)]))
        keys = avg + outputs
        logs_dict = OrderedDict()
        for key in keys:
            logs_dict[key] = []
        return logs_dict

    def on_train_begin(self, logs={}):
        self._time_start = time()
        s = []
        s.append('Epochs: %d' % (self.params['nb_epoch']))
        s.append('Samples: %d' % (self.params['nb_sample']))
        if hasattr(self, 'model'):
            tmp = 'Learning rate: %f' % (kback.eval(self.model.optimizer.lr))
            s.append(tmp)
        s = '\n'.join(s)
        self._log(s)

    def on_train_end(self, logs={}):
        self._log(self._line)

    def on_epoch_begin(self, epoch, logs={}):
        self._log(self._line)
        s = 'Epoch %d/%d' % (epoch + 1, self.params['nb_epoch'])
        self._log(s)
        self._log(self._line)
        self._nb_seen = 0
        self._nb_seen_freq = 0
        self._batch = 0
        self._nb_batch = None
        self._batch_logs = None
        self._totals = None

    def on_epoch_end(self, epoch, logs={}):
        if self._batch_logs:
            self.batch_logs.append(self._batch_logs)

        if not self.epoch_logs:
            self.epoch_logs = self._init_logs(logs)
            self.val_epoch_logs = self._init_logs(logs, False)

        for k, v in self.epoch_logs.items():
            if k in logs:
                v.append(logs[k])
            else:
                v.append(None)

        for k, v in self.val_epoch_logs.items():
            if 'val_' + k in logs:
                v.append(logs[k])
            else:
                v.append(None)

        table = OrderedDict()
        table['split'] = ['train']
        for k, v in self.epoch_logs.items():
            table[k] = [v[-1]]
        if self.val_epoch_logs:
            table['split'].append('val')
            for k, v in self.val_epoch_logs.items():
                table[k].append(v[-1])
        self._log('')
        self._log(format_table(table))

        for callback in self.callbacks:
            callback(epoch, self.epoch_logs, self.val_epoch_logs)

    def on_batch_end(self, batch, logs={}):
        self._batch += 1
        batch_size = logs.get('size', 0)
        self._nb_seen += batch_size
        if self._nb_batch is None:
            self._nb_batch = int(np.ceil(self.params['nb_sample'] / batch_size))

        if not self._batch_logs:
            self._batch_logs = self._init_logs(logs.keys())
            self._totals = OrderedDict()
            for k in self._batch_logs.keys():
                self._totals[k] = 0

        for k, v in logs.items():
            if k in self._totals:
                self._totals[k] += v * batch_size

        for k in self._batch_logs.keys():
            self._batch_logs[k].append(self._totals[k] / (self._nb_seen + 1e-5))

        do_log = False
        self._nb_seen_freq += batch_size
        if self._nb_seen_freq > int(self.params['nb_sample'] * self.log_freq):
            self.nb_seen_freq = 0
            do_log = True
        do_log = do_log or self._batch == 1 or self._nb_seen == self.params['nb_sample']

        if do_log:
            table = OrderedDict()
            table['progress (%)'] = [self._nb_seen / (self.params['nb_sample'] + 1e-5) * 100]
            table['time (min)'] = ['%.2f' % ((time() - self._time_start) / 60)]
            for k, v in self._batch_logs.items():
                table[k] = [v[-1]]
            self._log(format_table(table, header=self._batch==1))

class Timer(Callback):

    def __init__(self, max_time=None, verbose=1):
        """max_time in seconds."""
        self.max_time = max_time
        self.verbose = verbose

    def on_train_begin(self, logs={}):
        self._time_start = time()

    def on_epoch_end(self, batch, logs={}):
        if self.max_time is None:
            return
        elapsed = time() - self._time_start
        if elapsed > self.max_time:
            if self.verbose:
                print('Stop training after %.2fh' % (elapsed / 3600))
            self.model.stop_training = True
