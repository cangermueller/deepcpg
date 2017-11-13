"""Keras callback classes used by `dcpg_train.py`."""

from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import os
from time import time

from keras.callbacks import Callback

import numpy as np
import six

from .utils import format_table


class PerformanceLogger(Callback):
    """Logs performance metrics during training.

    Stores and prints performance metrics for each batch, epoch, and output.

    Parameters
    ----------
    metrics: list
        Name of metrics to be logged.
    log_freq: float
        Logging frequency as the percentage of training samples per epoch.
    precision: int
        Floating point precision.
    callbacks: list
        List of functions with parameters `epoch`, `epoch_logs`, and
        `val_epoch_logs` that are called at the end of each epoch.
    verbose: bool
        If `True`, log performance metrics of individual outputs.
    logger: function
        Logging function.
    """

    def __init__(self, metrics=['loss', 'acc'], log_freq=0.1,
                 precision=4, callbacks=[], verbose=bool, logger=print):
        self.metrics = metrics
        self.log_freq = log_freq
        self.precision = precision
        self.callbacks = callbacks
        self.verbose = verbose
        self.logger = logger
        self._line = '=' * 100
        self.epoch_logs = None
        self.val_epoch_logs = None
        self.batch_logs = []

    def _log(self, x):
        if self.logger:
            self.logger(x)

    def _init_logs(self, logs, train=True):
        """Extracts metric names from `logs` and initializes table to store
        epoch or batch logs.

        Returns
        -------
        tuple
            Tuple (`metrics`, `logs_dict`). `metrics` maps metrics, e.g.
            `metrics['acc'] = ['acc', 'output_acc1']`. `logs_dict` is a dict of
            lists to store logs, e.g.  `logs_dict['acc'] = []`.
        """

        logs = list(logs)
        # Select either only training or validation logs
        if train:
            logs = [log for log in logs if not log.startswith('val_')]
        else:
            logs = [log[4:] for log in logs if log.startswith('val_')]

        # `metrics` stores for each metric in self.metrics that exists in logs
        # the name for the metric itself, followed by all output metrics:
        #   metrics['acc'] = ['acc', 'output1_acc', 'output2_acc']
        metrics = OrderedDict()
        for name in self.metrics:
            if name in logs:
                metrics[name] = [name]
            output_logs = [log for log in logs if log.endswith('_' + name)]
            if len(output_logs):
                if name not in metrics:
                    # mean 'acc' does not exist in logs, but is added here to
                    # compute it later over all outputs with `_udpate_means`
                    metrics[name] = [name]
                metrics[name].extend(output_logs)

        # `logs_dict` stored the actual logs for each metric in `metrics`
        logs_dict = OrderedDict()
        # Show mean metrics first
        for mean_name in metrics:
            logs_dict[mean_name] = []
        # Followed by all output metrics
        for mean_name, names in six.iteritems(metrics):
            for name in names:
                logs_dict[name] = []

        return metrics, logs_dict

    def _update_means(self, logs, metrics):
        """Computes the mean over all outputs, if it does not exist yet."""

        for mean_name, names in six.iteritems(metrics):
            # Skip, if mean already exists, e.g. loss.
            if logs[mean_name][-1] is not None:
                continue
            mean = 0
            count = 0
            for name in names:
                if name in logs:
                    value = logs[name][-1]
                    if value is not None and not np.isnan(value):
                        mean += value
                        count += 1
            if count:
                mean /= count
            else:
                mean = np.nan
            logs[mean_name][-1] = mean

    def on_train_begin(self, logs={}):
        self._time_start = time()
        s = []
        s.append('Epochs: %d' % (self.params['epochs']))
        s = '\n'.join(s)
        self._log(s)

    def on_train_end(self, logs={}):
        self._log(self._line)

    def on_epoch_begin(self, epoch, logs={}):
        self._log(self._line)
        s = 'Epoch %d/%d' % (epoch + 1, self.params['epochs'])
        self._log(s)
        self._log(self._line)
        self._step = 0
        self._steps = self.params['steps']
        self._log_freq = int(np.ceil(self.log_freq * self._steps))
        self._batch_logs = None
        self._totals = None

    def on_epoch_end(self, epoch, logs={}):
        if self._batch_logs:
            self.batch_logs.append(self._batch_logs)

        if not self.epoch_logs:
            # Initialize epoch metrics and logs
            self._epoch_metrics, self.epoch_logs = self._init_logs(logs)
            tmp = self._init_logs(logs, False)
            self._val_epoch_metrics, self.val_epoch_logs = tmp

        # Add new epoch logs to logs table
        for metric, metric_logs in six.iteritems(self.epoch_logs):
            if metric in logs:
                metric_logs.append(logs[metric])
            else:
                # Add `None` if log value missing
                metric_logs.append(None)
        self._update_means(self.epoch_logs, self._epoch_metrics)

        # Add new validation epoch logs to logs table
        for metric, metric_logs in six.iteritems(self.val_epoch_logs):
            metric_val = 'val_' + metric
            if metric_val in logs:
                metric_logs.append(logs[metric_val])
            else:
                metric_logs.append(None)
        self._update_means(self.val_epoch_logs, self._val_epoch_metrics)

        # Show table
        table = OrderedDict()
        table['split'] = ['train']
        # Show mean logs first
        for mean_name in self._epoch_metrics:
            table[mean_name] = []
        # Show output logs
        if self.verbose:
            for mean_name, names in six.iteritems(self._epoch_metrics):
                for name in names:
                    table[name] = []
        for name, logs in six.iteritems(self.epoch_logs):
            if name in table:
                table[name].append(logs[-1])
        if self.val_epoch_logs:
            table['split'].append('val')
            for name, logs in six.iteritems(self.val_epoch_logs):
                if name in table:
                    table[name].append(logs[-1])
        self._log('')
        self._log(format_table(table, precision=self.precision))

        # Trigger callbacks
        for callback in self.callbacks:
            callback(epoch, self.epoch_logs, self.val_epoch_logs)

    def on_batch_end(self, batch, logs={}):
        self._step += 1
        batch_size = logs.get('size', 0)

        if not self._batch_logs:
            # Initialize batch metrics and logs table
            self._batch_metrics, self._batch_logs = self._init_logs(logs.keys())
            # Sum of logs up to the current batch
            self._totals = OrderedDict()
            # Number of samples up to the current batch
            self._nb_totals = OrderedDict()
            for name in self._batch_logs:
                if name in logs:
                    self._totals[name] = 0
                    self._nb_totals[name] = 0

        for name, value in six.iteritems(logs):
            # Skip value if nan, which can occur if the batch size is small.
            if np.isnan(value):
                continue
            if name in self._totals:
                self._totals[name] += value * batch_size
                self._nb_totals[name] += batch_size

        # Compute the accumulative mean over logs and store it in `_batch_logs`.
        for name in self._batch_logs:
            if name in self._totals:
                if self._nb_totals[name]:
                    tmp = self._totals[name] / self._nb_totals[name]
                else:
                    tmp = np.nan
            else:
                tmp = None
            self._batch_logs[name].append(tmp)
        self._update_means(self._batch_logs, self._batch_metrics)

        # Show logs table at a certain frequency
        do_log = False
        if self._step % self._log_freq == 0:
            do_log = True
        do_log |= self._step == 1 or self._step == self._steps

        if do_log:
            table = OrderedDict()
            prog = self._step / self._steps
            prog *= 100
            precision = []
            table['done (%)'] = [prog]
            precision.append(1)
            table['time'] = [(time() - self._time_start) / 60]
            precision.append(1)
            for mean_name in self._batch_metrics:
                table[mean_name] = []
            if self.verbose:
                for mean_name, names in six.iteritems(self._batch_metrics):
                    for name in names:
                        table[name] = []
                        precision.append(self.precision)
            for name, logs in six.iteritems(self._batch_logs):
                if name in table:
                    table[name].append(logs[-1])
                    precision.append(self.precision)

            self._log(format_table(table, precision=precision,
                                   header=self._step == 1))


class TrainingStopper(Callback):
    """Stop training after certain time or when file is detected.

    Parameters
    ----------
    max_time: int
        Maximum training time in seconds.
    stop_file: str
        Name of stop file that triggers the end of training when existing.
    verbose: bool
        If `True`, log message when training is stopped.
    """

    def __init__(self, max_time=None, stop_file=None,
                 verbose=1, logger=print):
        """max_time in seconds."""
        self.max_time = max_time
        self.stop_file = stop_file
        self.verbose = verbose
        self.logger = logger

    def on_train_begin(self, logs={}):
        self._time_start = time()

    def log(self, msg):
        if self.verbose:
            self.logger(msg)

    def on_epoch_end(self, batch, logs={}):
        if self.max_time is not None:
            elapsed = time() - self._time_start
            if elapsed > self.max_time:
                self.log('Stopping training after %.2fh' % (elapsed / 3600))
                self.model.stop_training = True

        if self.stop_file:
            if os.path.isfile(self.stop_file):
                self.log('Stopping training due to stop file!')
                self.model.stop_training = True
