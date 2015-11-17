import sys
from keras.callbacks import Callback
import pandas as pd
import numpy as np
import warnings

from utils import MASK
from predict.evaluation import eval_funs


class PerformanceLogger(Callback):

    def __init__(self, data, validation_data=None):
        self.data = data
        self.validation_data = validation_data
        self.eval_funs = eval_funs
        self.eval_names = [x[0] for x in self.eval_funs]
        self.logs = dict()
        self.logs['train'] = None
        if validation_data is not None:
            self.logs['val'] = None

    def stack(self, epoch=None):
        l = sorted(self.logs.keys())
        d = [self.logs[k] for k in l]
        if epoch:
            d = [x.loc[epoch] for x in d]
        d = pd.concat(d, axis=1, keys=l)
        return d

    def on_epoch_end(self, epoch, logs={}):
        self.logs['train'] = self.add(self.data, self.logs['train'], epoch)
        if self.validation_data:
            self.logs['val'] = self.add(self.validation_data, self.logs['val'], epoch)
        d = self.stack(epoch)
        print(np.round(d, 4))

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


class ModelCheckpoint(Callback):

    def __init__(self, filepath, monitor='val_loss', verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs={}):
        self.model.save_weights(self.filepath + '_last.h5', overwrite=True)
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Can save best model only with %s available, skipping." % (self.monitor), RuntimeWarning)
        else:
            if current < self.best:
                if self.verbose > 0:
                    print("Epoch %05d: %s improved from %0.5f to %0.5f, saving model to %s"
                          % (epoch, self.monitor, self.best, current, self.filepath))
                self.best = current
                self.model.save_weights(self.filepath + '_best.h5', overwrite=True)
            else:
                if self.verbose > 0:
                    print("Epoch %05d: %s did not improve" % (epoch, self.monitor))
            sys.stdout.flush()
