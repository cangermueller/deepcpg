from collections import OrderedDict
import os
from pkg_resources import parse_version
from time import time

from keras import backend as K
from keras.callbacks import Callback

import numpy as np

from .utils import format_table, EPS


class PerformanceLogger(Callback):

    def __init__(self, metrics=['loss', 'acc'], log_freq=0.1,
                 precision=4, callbacks=[], verbose=1, logger=print):
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
        logs = list(logs)
        if train:
            logs = [log for log in logs if not log.startswith('val_')]
        else:
            logs = [log[4:] for log in logs if log.startswith('val_')]

        metrics = OrderedDict()
        for name in self.metrics:
            if name in logs:
                metrics[name] = [name]
            output_logs = [log for log in logs if log.endswith('_' + name)]
            if len(output_logs):
                if name not in metrics:
                    metrics[name] = [name]
                metrics[name].extend(output_logs)

        logs_dict = OrderedDict()
        for mean_name, names in metrics.items():
            for name in names:
                logs_dict[name] = []

        return metrics, logs_dict

    def _update_means(self, logs, metrics):
        for mean_name, names in metrics.items():
            if logs[mean_name][-1] is not None:
                continue
            mean = 0
            count = 0
            for name in names:
                if name in logs:
                    if logs[name][-1] is not None:
                        mean += logs[name][-1]
                        count += 1
            mean /= count + EPS
            logs[mean_name][-1] = mean

    def on_train_begin(self, logs={}):
        self._time_start = time()
        s = []
        s.append('Epochs: %d' % (self.params['nb_epoch']))
        s.append('Samples: %d' % (self.params['nb_sample']))
        if hasattr(self, 'model'):
            lr = K.eval(self.model.optimizer.lr)
            lr = float(np.asarray(lr))
            s.append('Learning rate: %f' % (lr))
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
            self._epoch_metrics, self.epoch_logs = self._init_logs(logs)
            tmp = self._init_logs(logs, False)
            self._val_epoch_metrics, self.val_epoch_logs = tmp

        for k, v in self.epoch_logs.items():
            if k in logs:
                v.append(logs[k])
            else:
                v.append(None)
        self._update_means(self.epoch_logs, self._epoch_metrics)

        for k, v in self.val_epoch_logs.items():
            k_val = 'val_' + k
            if k_val in logs:
                v.append(logs[k_val])
            else:
                v.append(None)
        self._update_means(self.val_epoch_logs, self._val_epoch_metrics)

        table = OrderedDict()
        table['split'] = ['train']
        for k, v in self.epoch_logs.items():
            if self.verbose or k in self._epoch_metrics:
                table[k] = [v[-1]]
        if self.val_epoch_logs:
            table['split'].append('val')
            for k, v in self.val_epoch_logs.items():
                if self.verbose or k in self._val_epoch_metrics:
                    table[k].append(v[-1])
        self._log('')
        self._log(format_table(table, precision=self.precision))

        for callback in self.callbacks:
            callback(epoch, self.epoch_logs, self.val_epoch_logs)

    def on_batch_end(self, batch, logs={}):
        self._batch += 1
        batch_size = logs.get('size', 0)
        self._nb_seen += batch_size
        if self._nb_batch is None:
            self._nb_batch = int(np.ceil(self.params['nb_sample'] /
                                         (batch_size + EPS)))

        if not self._batch_logs:
            self._batch_metrics, self._batch_logs = self._init_logs(logs.keys())
            self._totals = OrderedDict()
            for k in self._batch_logs.keys():
                if k in logs:
                    self._totals[k] = 0

        for k, v in logs.items():
            if k in self._totals:
                self._totals[k] += v * batch_size

        for k in self._batch_logs.keys():
            if k in self._totals:
                tmp = self._totals[k] / (self._nb_seen + EPS)
            else:
                tmp = None
            self._batch_logs[k].append(tmp)
        self._update_means(self._batch_logs, self._batch_metrics)

        do_log = False
        self._nb_seen_freq += batch_size
        if self._nb_seen_freq > int(self.params['nb_sample'] * self.log_freq):
            self.nb_seen_freq = 0
            do_log = True
        do_log |= self._batch == 1 or self._nb_seen == self.params['nb_sample']

        if do_log:
            table = OrderedDict()
            prog = self._nb_seen / (self.params['nb_sample'] + EPS)
            prog *= 100
            precision = []
            table['progress (%)'] = [prog]
            precision.append(1)
            table['time (min)'] = [(time() - self._time_start) / 60]
            precision.append(1)

            for name, logs in self._batch_logs.items():
                if self.verbose or name in self._batch_metrics:
                    table[name] = [self._batch_logs[name][-1]]
                    precision.append(self.precision)

            self._log(format_table(table, precision=precision,
                                   header=self._batch == 1))
            self._nb_seen_freq = 0


class TrainingStopper(Callback):

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


class TensorBoard(Callback):
    ''' Tensorboard basic visualizations.

    This callback writes a log for TensorBoard, which allows
    you to visualize dynamic graphs of your training and test
    metrics, as well as activation histograms for the different
    layers in your model.

    TensorBoard is a visualization tool provided with TensorFlow.

    If you have installed TensorFlow with pip, you should be able
    to launch TensorBoard from the command line:
    ```
    tensorboard --logdir=/full_path_to_your_logs
    ```
    You can find more information about TensorBoard
    [here](https://www.tensorflow.org/versions/master/how_tos/summaries_and_tensorboard/index.html).

    # Arguments
        log_dir: the path of the directory where to save the log
            files to be parsed by Tensorboard
        histogram_freq: frequency (in epochs) at which to compute activation
            histograms for the layers of the model. If set to 0,
            histograms won't be computed.
        write_graph: whether to visualize the graph in Tensorboard.
            The log file can become quite large when
            write_graph is set to True.
    '''

    def __init__(self, log_dir='./logs', histogram_freq=0, write_graph=True,
                 write_images=False):
        super(TensorBoard, self).__init__()
        if K._BACKEND != 'tensorflow':
            raise Exception('TensorBoard callback only works '
                            'with the TensorFlow backend.')
        self.log_dir = log_dir
        self.histogram_freq = histogram_freq
        self.merged = None
        self.write_graph = write_graph
        self.write_images = write_images

    def _set_model(self, model):
        import tensorflow as tf
        import keras.backend.tensorflow_backend as KTF

        self.model = model
        self.sess = KTF.get_session()
        if self.histogram_freq and self.merged is None:
            for layer in self.model.layers:

                for weight in layer.weights:
                    tf.histogram_summary(weight.name, weight)

                    if self.write_images:
                        w_img = tf.squeeze(weight)

                        shape = w_img.get_shape()
                        if len(shape) > 1 and shape[0] > shape[1]:
                            w_img = tf.transpose(w_img)

                        if len(shape) == 1:
                            w_img = tf.expand_dims(w_img, 0)

                        w_img = tf.expand_dims(tf.expand_dims(w_img, 0), -1)

                        tf.image_summary(weight.name, w_img)

                if hasattr(layer, 'output'):
                    tf.histogram_summary('{}_out'.format(layer.name),
                                         layer.output)
        self.merged = tf.merge_all_summaries()
        if self.write_graph:
            if parse_version(tf.__version__) >= parse_version('0.8.0'):
                self.writer = tf.train.SummaryWriter(self.log_dir,
                                                     self.sess.graph)
            else:
                self.writer = tf.train.SummaryWriter(self.log_dir,
                                                     self.sess.graph_def)
        else:
            self.writer = tf.train.SummaryWriter(self.log_dir)

    def on_epoch_end(self, epoch, logs={}):
        import tensorflow as tf

        if self.model.validation_data and self.histogram_freq:
            if epoch % self.histogram_freq == 0:
                # TODO: implement batched calls to sess.run
                # (current call will likely go OOM on GPU)
                if self.model.uses_learning_phase:
                    cut_v_data = len(self.model.inputs)
                    val_data = self.model.validation_data[:cut_v_data] + [0]
                    tensors = self.model.inputs + [K.learning_phase()]
                else:
                    val_data = self.model.validation_data
                    tensors = self.model.inputs
                feed_dict = dict(zip(tensors, val_data))
                result = self.sess.run([self.merged], feed_dict=feed_dict)
                summary_str = result[0]
                self.writer.add_summary(summary_str, epoch)

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            self.writer.add_summary(summary, epoch)
        self.writer.flush()
