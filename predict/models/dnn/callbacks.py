from keras.callbacks import Callback
import pandas as pd
import numpy as np
from time import time
import warnings


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


class LearningRateScheduler(Callback):

    def __init__(self, callback, monitor='val_loss', patience=0):
        super(LearningRateScheduler, self).__init__()
        self.callback = callback
        self.patience = patience
        self.monitor = monitor

        self.counter = 0
        self.best_score = np.inf
        self.best_weights = None

    def on_epoch_end(self, epoch, logs={}):
        score = logs.get(self.monitor)
        if self.best_weights is None or score <= self.best_score:
            self.counter = 0
            self.best_score = score
            self.best_weights = self.model.get_weights()
        else:
            self.counter += 1
            if self.counter > self.patience:
                self.callback()
                self.model.set_weights(self.best_weights)
                self.counter = 0


class PerformanceLogger(Callback):

    def __init__(self, batch_logs=['loss', 'acc'],
                 epoch_logs=['val_loss', 'val_acc'], callbacks=[]):
        if batch_logs is None:
            batch_logs = []
        if epoch_logs is None:
            epoch_logs = []
        self.batch_logs = batch_logs
        self.epoch_logs = epoch_logs
        self.callbacks = callbacks

    def on_train_begin(self, logs={}):
        self._batch_logs = []
        self._epoch_logs = []

    def on_epoch_begin(self, epoch, logs={}):
        self._batch_logs.append([])

    def on_batch_end(self, batch, logs={}):
        l = {k: v for k, v in logs.items() if k in self.batch_logs}
        self._batch_logs[-1].append(l)

    def on_epoch_end(self, batch, logs={}):
        l = {k: v for k, v in logs.items() if k in self.epoch_logs}
        self._epoch_logs.append(l)
        for cb in self.callbacks:
            cb()

    def _list_to_frame(self, l, keys):
        keys = [k for k in keys if k in l[0].keys()]
        d = {k: [] for k in keys}
        for ll in l:
            for k in keys:
                d[k].append(float(ll[k]))
        d = pd.DataFrame(d, columns=keys)
        return d

    def epoch_frame(self):
        d = self._list_to_frame(self._epoch_logs, self.epoch_logs)
        t = list(d.columns)
        d['epoch'] = np.arange(d.shape[0]) + 1
        d = d.loc[:, ['epoch'] + t]
        return d

    def batch_frame(self, epoch=None):
        if epoch is None:
            d = []
            for e in range(len(self._batch_logs)):
                de = self.batch_frame(e + 1)
                t = list(de.columns)
                de['epoch'] = e + 1
                de = de.loc[:, ['epoch'] + t]
                d.append(de)
            d = pd.concat(d)
        else:
            d = self._list_to_frame(self._batch_logs[epoch - 1],
                                    self.batch_logs)
            t = list(d.columns)
            d['batch'] = np.arange(d.shape[0]) + 1
            d = d.loc[:, ['batch'] + t]
        return d

    def frame(self):
        b = self.batch_frame().groupby('epoch', as_index=False).mean()
        b = b.loc[:, b.columns != 'batch']
        e = self.epoch_frame()
        c = pd.merge(b, e, on='epoch')
        return c


class ProgressLogger(Callback):

    def __init__(self,
                 batch_logs=['loss', 'acc'],
                 epoch_logs=['val_loss', 'val_acc'],
                 interval=0.1,
                 logger=print):
        if batch_logs is None:
            batch_logs = []
        if epoch_logs is None:
            epoch_logs = []
        self.batch_logs = batch_logs
        self.epoch_logs = epoch_logs
        self.interval = interval
        self.logger = logger
        self._line = '-' * 80

    def _log(self, x):
        self.logger(x)

    def on_train_begin(self, logs={}):
        self._time_start = time()
        s = []
        s.append('Epochs: %d' % (self.params['nb_epoch']))
        s.append('Samples: %d' % (self.params['nb_sample']))
        s.append('Batch size: %d' % (self.params['batch_size']))
        if hasattr(self, 'model'):
            s.append('Learning rate: %f' % (
                self.model.optimizer.lr.get_value()))
        s = '\n'.join(s)
        self._log(s)

    def on_epoch_begin(self, epoch, logs={}):
        self._nb_batch = int(np.ceil(self.params['nb_sample'] /
                                     self.params['batch_size']))
        self._batch = 0
        self._interval = max(1, round(self._nb_batch * self.interval))
        s = 'Epoch %d/%d' % (epoch + 1, self.params['nb_epoch'])
        self._log(self._line)
        self._log(s)
        self._log(self._line)
        self._seen = 0
        self._totals = dict()

    def on_batch_end(self, batch, logs={}):
        self._batch += 1
        batch_size = logs.get('size', 0)
        self._seen += batch_size

        for k, v in logs.items():
            if k in self.batch_logs:
                if k not in self._totals.keys():
                    self._totals[k] = 0
                self._totals[k] += v * batch_size

        if self._batch == 1 or self._batch % self._interval == 0:
            mins = (time() - self._time_start) / 60
            t = len(str(self._nb_batch))
            s = '%5.1f%%\t(%' + str(t) + 'd/%d)\t%7.2f min'
            s = s % (
                self._batch / (self._nb_batch + 1e-5) * 100,
                self._batch,
                self._nb_batch,
                mins
            )
            for k, v in self._totals.items():
                s += '\t%s=%.3f' % (k, v / max(self._seen, 1e-5))
            self._log(s)

    def on_epoch_end(self, batch, logs={}):
        s = ''
        for k, v in self._totals.items():
            s += '\t%s=%.3f' % (k, v / max(self._seen, 1e-5))
        for k in self.epoch_logs:
            if k in logs:
                s += '\t%5s=%.3f' % (k, logs[k])
        s = s.strip()
        if len(s):
            self._log(self._line)
            self._log(s)


class DataJumper(Callback):

    def __init__(self, data, nb_sample=None, verbose=0, start=0, stop=None,
                 jump=True):
        self.data = data
        self._n = self.data[0].shape[0]
        if nb_sample is None:
            nb_sample = self._n
        else:
            nb_sample = min(self._n, nb_sample)
        self.nb_sample = nb_sample
        self.verbose = verbose
        if stop is None:
            stop = start + self.nb_sample
        for d in data:
            d.start = start
            d.stop = stop
        self.jump = jump

    def on_epoch_begin(self, epoch, logs={}):
        if self.jump and self._n != self.nb_sample:
            i = np.random.randint(self._n - self.nb_sample)
            for d in self.data:
                d.start = i
                d.stop = i + self.nb_sample
        if self.verbose:
            d = self.data[0]
            print('Index: (%d - %d)' % (d.start, d.stop))


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
