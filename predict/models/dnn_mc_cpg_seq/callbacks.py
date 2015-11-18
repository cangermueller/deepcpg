from keras.callbacks import Callback
import pandas as pd
import numpy as np

from utils import MASK
from predict.evaluation import eval_funs


class LearningRateScheduler(Callback):

    def __init__(self, callback, monitor='val_loss', patience=0):
        super(LearningRateScheduler, self).__init__()
        self.callback = callback
        self.patience = patience
        self.monitor = monitor

        self.counter = 0
        self.prev_score = np.inf
        self.best_score = np.inf
        self.best_weights = None

    def on_epoch_end(self, epoch, logs={}):
        score = logs.get(self.monitor)
        if score <= self.prev_score:
            self.counter = 0
            if score <= self.best_score:
                self.best_score = score
                self.best_weights = self.model.get_weights()
        else:
            self.counter += 1
            if self.counter > self.patience:
                self.callback()
                self.model.set_weights(self.best_weights)
                self.counter = 0
        self.prev_score = score


class PerformanceLogger(Callback):

    def __init__(self, data, label=None):
        self.data = data
        self.eval_funs = eval_funs
        self.eval_names = [x[0] for x in self.eval_funs]
        self.logs = None
        self.label = label

    def on_epoch_end(self, epoch, logs={}):
        self.logs = self.add(self.data, self.logs, epoch)
        d = self.logs.loc[epoch]
        if self.label is None:
            t = 'Performance:'
        else:
            t = 'Performance (%s):' % (self.label)
        print(t)
        print(np.round(d, 4))
        print()

    def add(self, data, logs, epoch):
        log = self.evaluate(data, epoch)
        if logs is None:
            logs = log
        else:
            logs = pd.concat((logs, log))
        return logs

    def evaluate(self, data, epoch):
        log = {k: [] for k, f in self.eval_funs}
        X = {k: data[k] for k in self.model.input_order}
        zall = self.model.predict(X)
        for k in self.model.output_order:
            y = data[k]
            z = zall[k].ravel()
            t = y != MASK
            y = y[t]
            z = z[t]
            for eval_fun in self.eval_funs:
                log[eval_fun[0]].append(eval_fun[1](y, z))
        i = pd.MultiIndex.from_product([[epoch], self.model.output_order])
        log = pd.DataFrame(log, columns=self.eval_names,
                           index=i)
        return log

    def __str__(self):
        return str(self.stack())
