import numpy as np
from time import sleep

from keras.callbacks import CallbackList

import predict.models.dnn.callbacks as cbk


def test_progress_logger():
    batch_logs = ['loss', 'acc']
    epoch_logs = ['val_loss', 'val_acc']
    callbacks = [cbk.ProgressLogger()]
    callbacks = CallbackList(callbacks)
    params = {
        'batch_size': 128,
        'nb_epoch': 3,
        'nb_sample': 10000
    }
    callbacks._set_params(params)
    callbacks.on_train_begin()
    for epoch in range(params['nb_epoch']):
        callbacks.on_epoch_begin(epoch)
        for i in range(0, params['nb_sample'], params['batch_size']):
            sleep(0.0)
            callbacks.on_batch_begin(i)
            logs = {k: np.random.rand() for k in batch_logs}
            callbacks.on_batch_end(i, logs)
        logs = {k: np.random.rand() for k in epoch_logs}
        callbacks.on_epoch_end(epoch, logs)
    callbacks.on_train_end()
