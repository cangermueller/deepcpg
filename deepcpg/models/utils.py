"""Functions for building, training, and loading models.
"""

from __future__ import division
from __future__ import print_function

from os import path as pt

from keras import backend as K
from keras import models as km
from keras import layers as kl
from keras.utils.np_utils import to_categorical
import numpy as np
import pandas as pd

from .. import data as dat
from .. import evaluation as ev
from ..data import hdf, OUTPUT_SEP
from ..data.dna import int_to_onehot
from ..utils import to_list


class ScaledSigmoid(kl.Layer):
    """Scaled sigmoid activation function.

    Scales the maximum of the sigmoid function from one to the provided value.

    Parameters
    ----------
    scaling: float
        Maximum of sigmoid function.
    """
    def __init__(self, scaling=1.0, **kwargs):
        self.supports_masking = True
        self.scaling = scaling
        super(ScaledSigmoid, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return K.sigmoid(x) * self.scaling

    def get_config(self):
        config = {'scaling': self.scaling}
        base_config = super(ScaledSigmoid, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


CUSTOM_OBJECTS = {'ScaledSigmoid': ScaledSigmoid}


def get_first_conv_layer(layers, get_act=False):
    """Return the first convolutional layers in a stack of layer.

    Parameters
    ----------
    layers: list
        List of Keras layers.
    get_act: bool
        Return the activation layer after the convolutional weight layer.

    Returns
    -------
    Keras layer
        Convolutional layer or tuple of convolutional layer and activation layer
        if `get_act=True`.
    """
    conv_layer = None
    act_layer = None
    for layer in layers:
        if isinstance(layer, kl.Conv1D) and layer.input_shape[-1] == 4:
            conv_layer = layer
            if not get_act:
                break
        elif conv_layer and isinstance(layer, kl.Activation):
            act_layer = layer
            break
    if not conv_layer:
        raise ValueError('Convolutional layer not found')
    if get_act:
        if not act_layer:
            raise ValueError('Activation layer not found')
        return (conv_layer, act_layer)
    else:
        return conv_layer


def get_sample_weights(y, class_weights=None):
    """Compute sample weights for model training.

    Computes sample weights given  a vector of output labels `y`. Sets weights
    of samples without label (`CPG_NAN`) to zero.

    Parameters
    ----------
    y: :class:`numpy.ndarray`
        1d numpy array of output labels.
    class_weights: dict
        Weight of output classes, e.g. methylation states.

    Returns
    -------
    :class:`numpy.ndarray`
        Sample weights of size `y`.
    """
    y = y[:]
    sample_weights = np.ones(y.shape, dtype=K.floatx())
    sample_weights[y == dat.CPG_NAN] = K.epsilon()
    if class_weights is not None:
        for cla, weight in class_weights.items():
            sample_weights[y == cla] = weight
    return sample_weights


def save_model(model, model_file, weights_file=None):
    """Save Keras model to file.

    If `model_file` ends with '.h5', saves model description and model weights
    in HDF5 file. Otherwise, saves JSON model description in `model_file`
    and model weights in `weights_file` if provided.

    Parameters
    ----------
    model
        Keras model.
    model_file: str
        Output file.
    weights_file: str
        Weights file.
    """

    if pt.splitext(model_file)[1] == '.h5':
        model.save(model_file)
    else:
        with open(model_file, 'w') as f:
            f.write(model.to_json())
    if weights_file is not None:
        model.save_weights(weights_file, overwrite=True)


def search_model_files(dirname):
    """Search model files in given directory.

    Parameters
    ----------
    dirname: str
        Directory name

    Returns
    -------
    Model JSON file and weights if existing, otherwise HDF5 file.  None if no
    model files could be found.
    """

    json_file = pt.join(dirname, 'model.json')
    if pt.isfile(json_file):
        order = ['model_weights.h5', 'model_weights_val.h5',
                 'model_weights_train.h5']
        for name in order:
            filename = pt.join(dirname, name)
            if pt.isfile(filename):
                return [json_file, filename]
    elif pt.isfile(pt.join(dirname, 'model.h5')):
        return pt.join(dirname, 'model.h5')
    else:
        return None


def load_model(model_files, custom_objects=CUSTOM_OBJECTS, log=None):
    """Load Keras model from a list of model files.

    Loads Keras model from list of filenames, e.g. from `search_model_files`.
    `model_files` can be single HDF5 file, or JSON and weights file.

    Parameters
    ----------
    model_file: list
        Input model file names.
    custom_object: dict
        Custom objects for loading models that were trained with custom objects,
        e.g. `ScaledSigmoid`.

    Returns
    -------
    Keras model.
    """
    if not isinstance(model_files, list):
        model_files = [model_files]
    if pt.isdir(model_files[0]):
        model_files = search_model_files(model_files[0])
        if model_files is None:
            raise ValueError('No model found in "%s"!' % model_files[0])
        if log:
            log('Using model files %s' % ' '.join(model_files))
    if pt.splitext(model_files[0])[1] == '.h5':
        model = km.load_model(model_files[0], custom_objects=custom_objects)
    else:
        with open(model_files[0], 'r') as f:
            model = f.read()
        model = km.model_from_json(model, custom_objects=custom_objects)
    if len(model_files) > 1:
        model.load_weights(model_files[1])
    return model


def get_objectives(output_names):
    """Return training objectives for a list of output names.

    Returns
    -------
    dict
        dict with `output_names` as keys and the name of the assigned Keras
        objective as values.
    """
    objectives = dict()
    for output_name in output_names:
        _output_name = output_name.split(OUTPUT_SEP)
        if _output_name[0] in ['bulk']:
            objective = 'mean_squared_error'
        elif _output_name[-1] in ['mean', 'var']:
            objective = 'mean_squared_error'
        elif _output_name[-1] in ['cat_var']:
            objective = 'categorical_crossentropy'
        else:
            objective = 'binary_crossentropy'
        objectives[output_name] = objective
    return objectives


def add_output_layers(stem, output_names, init='glorot_uniform'):
    """Add and return outputs to a given layer.

    Adds output layer for each output in `output_names` to layer `stem`.

    Parameters
    ----------
    stem: Keras layer
        Keras layer to which output layers are added.
    output_names: list
        List of output names.

    Returns
    -------
    list
        Output layers added to `stem`.
    """
    outputs = []
    for output_name in output_names:
        _output_name = output_name.split(OUTPUT_SEP)
        if _output_name[-1] in ['entropy']:
            x = kl.Dense(1, kernel_initializer=init, activation='relu')(stem)
        elif _output_name[-1] in ['var']:
            x = kl.Dense(1, kernel_initializer=init)(stem)
            x = ScaledSigmoid(0.251, name=output_name)(x)
        elif _output_name[-1] in ['cat_var']:
            x = kl.Dense(3, kernel_initializer=init,
                         activation='softmax',
                         name=output_name)(stem)
        else:
            x = kl.Dense(1, kernel_initializer=init,
                         activation='sigmoid',
                         name=output_name)(stem)
        outputs.append(x)
    return outputs


def predict_generator(model, generator, nb_sample=None):
    """Predict model outputs using generator.

    Calls `model.predict` for at most `nb_sample` samples from `generator`.

    Parameters
    ----------
    model: Keras model
        Model to be evaluated.
    generator: generator
        Data generator.
    nb_sample: int
        Maximum number of samples.

    Returns
    -------
    list
        list [`inputs`, `outputs`, `predictions`].
    """
    data = None
    nb_seen = 0
    for data_batch in generator:
        if not isinstance(data_batch, list):
            data_batch = list(data_batch)

        if nb_sample:
            # Reduce batch size if needed
            nb_left = nb_sample - nb_seen
            for data_item in data_batch:
                for key, value in data_item.items():
                    data_item[key] = data_item[key][:nb_left]

        preds = model.predict(data_batch[0])
        if not isinstance(preds, list):
            preds = [preds]
        preds = {name: pred for name, pred in zip(model.output_names, preds)}

        if not data:
            data = [dict() for i in range(len(data_batch))]
        dat.add_to_dict(preds, data[0])
        for i in range(1, len(data_batch)):
            dat.add_to_dict(data_batch[i], data[i])

        nb_seen += len(list(preds.values())[0])
        if nb_sample and nb_seen >= nb_sample:
            break

    for i in range(len(data)):
        data[i] = dat.stack_dict(data[i])
    return data


def evaluate_generator(model, generator, return_data=False, *args, **kwargs):
    """Evaluate model on generator.

    Uses `predict_generator` to obtain predictions and `ev.evaluate` to evaluate
    predictions.

    Parameters
    ----------
    model
        Model to be evaluated.
    generator
        Data generator.
    return_rate: bool
        Return predictions and labels.
    *args: list
        Unnamed arguments passed to `predict_generator`.
    *kwargs: dict
        Named arguments passed to `predict_generator`.

    Returns
    -------
    If `return_data=False`, pandas data frame with performance metrics. If
    `return_data=True`, tuple (`perf`, `data`) with performance metrics `perf`
    and `data`.
    """
    data = predict_generator(model, generator, *args, **kwargs)
    perf = []
    for output in model.output_names:
        tmp = ev.evaluate(data[1][output], data[0][output])
        perf.append(pd.DataFrame(tmp, index=[output]))
    perf = pd.concat(perf)
    if return_data:
        return (perf, data)
    else:
        return perf


def read_from(reader, nb_sample=None):
    """Read `nb_sample` samples from `reader`."""
    data = None
    nb_seen = 0
    for data_batch in reader:
        if not isinstance(data_batch, list):
            data_batch = list(data_batch)

        if not data:
            data = [dict() for i in range(len(data_batch))]
        for i in range(len(data_batch)):
            dat.add_to_dict(data_batch[i], data[i])

        nb_seen += len(list(data_batch[0].values())[0])
        if nb_sample and nb_seen >= nb_sample:
            break

    for i in range(len(data)):
        data[i] = dat.stack_dict(data[i])
        if nb_sample:
            for key, value in data[i].items():
                data[i][key] = value[:nb_sample]
    return data


def copy_weights(src_model, dst_model, must_exist=True):
    """Copy weights from `src_model` to `dst_model`.

    Parameters
    ----------
    src_model
        Keras source model.
    dst_model
        Keras destination model.
    must_exist: bool
        If `True`, raises `ValueError` if a layer in `dst_model` does not exist
        in `src_model`.

    Returns
    -------
    list
        Names of layers that were copied.
    """
    copied = []
    for dst_layer in dst_model.layers:
        for src_layer in src_model.layers:
            if src_layer.name == dst_layer.name:
                break
        if not src_layer:
            if must_exist:
                tmp = 'Layer "%s" not found!' % (src_layer.name)
                raise ValueError(tmp)
            else:
                continue
        dst_layer.set_weights(src_layer.get_weights())
        copied.append(dst_layer.name)
    return copied


class Model(object):
    """Abstract model call.

    Abstract class of DNA, CpG, and Joint models.

    Parameters
    ----------
    dropout: float
        Dropout rate.
    l1_decay: float
        L1 weight decay.
    l2_decay: float
        L2 weight decay.
    init: str
        Name of Keras initialization.
    """

    def __init__(self, dropout=0.0, l1_decay=0.0, l2_decay=0.0,
                 init='glorot_uniform'):
        self.dropout = dropout
        self.l1_decay = l1_decay
        self.l2_decay = l2_decay
        self.init = init
        self.name = self.__class__.__name__
        self.scope = None

    def inputs(self, *args, **kwargs):
        """Return list of Keras model inputs."""
        pass

    def _build(self, input, output):
        """Build final model at the end of `__call__`."""
        model = km.Model(input, output, name=self.name)
        if self.scope:
            for layer in model.layers:
                if layer not in model.input_layers:
                    layer.name = '%s/%s' % (self.scope, layer.name)
        return model

    def __call__(self, inputs=None):
        """Build model.

        Parameters
        ----------
        inputs: list
            Keras model inputs
        """
        pass


def encode_replicate_names(replicate_names):
    """Encode list of replicate names as single string.

    .. note:: Deprecated
        This function is used to support legacy models and will be removed in
        the future.
    """
    return '--'.join(replicate_names)


def decode_replicate_names(replicate_names):
    """Decode string of replicate names and return names as list.

    .. note:: Deprecated
        This function is used to support legacy models and will be removed in
        the future.
    """
    return replicate_names.split('--')


class DataReader(object):
    """Read data from `dcpg_data.py` output files.

    Generator to read data batches from `dcpg_data.py` output files. Reads data
    using :func:`hdf.reader` and pre-processes data.

    Parameters
    ----------
    output_names: list
        Names of outputs to be read.
    use_dna: bool
        If `True`, read DNA sequence windows.
    dna_wlen: int
        Maximum length of DNA sequence windows.
    replicate_names: list
        Name of cells (profiles) whose neighboring CpG sites are read.
    cpg_wlen: int
        Maximum number of neighboring CpG sites.
    cpg_max_dist: int
        Value to threshold the distance of neighboring CpG sites.
    encode_replicates: bool
        If `True`, encode replicated names in key of returned dict. This option
        is deprecated and will be removed in the future.

    Returns
    -------
    tuple
        `dict` (`inputs`, `outputs`, `weights`), where `inputs`, `outputs`,
        `weights` is a `dict` of model inputs, outputs, and output weights.
        `outputs` and `weights` are not returned if `output_names` is undefined.
    """
    def __init__(self, output_names=None,
                 use_dna=True, dna_wlen=None,
                 replicate_names=None, cpg_wlen=None, cpg_max_dist=25000,
                 encode_replicates=False):
        self.output_names = to_list(output_names)
        self.use_dna = use_dna
        self.dna_wlen = dna_wlen
        self.replicate_names = to_list(replicate_names)
        self.cpg_wlen = cpg_wlen
        self.cpg_max_dist = cpg_max_dist
        self.encode_replicates = encode_replicates

    def _prepro_dna(self, dna):
        """Preprocess DNA sequence windows.

        Slices DNA sequence window if `self.dna_wlen` is defined and one-hot
        encodes sequences.

        Parameters
        ----------
        dna: :class:`numpy.ndarray`
            :class:`numpy.ndarray` of size [nb_window, window_len] with integer
            sequences windows.

        Returns
        -------
        :class:`numpy.ndarray`
            :class:`numpy.ndarray` of size [nb_window, window_len, 4] with
            one-hot encoded sequences.
        """
        if self.dna_wlen:
            cur_wlen = dna.shape[1]
            center = cur_wlen // 2
            delta = self.dna_wlen // 2
            dna = dna[:, (center - delta):(center + delta + 1)]
        return int_to_onehot(dna)

    def _prepro_cpg(self, states, dists):
        """Preprocess the state and distance of neighboring CpG sites.

        Parameters
        ----------
        states: list
            List of CpG states of all replicates.
        dists: list
            List of CpG distances of all replicates.

        Returns
        -------
        prepro_states: list
            List of preprocessed CpG states of all replicates.
        prepro_dists: list
            List of preprocessed CpG distances of all replicates.
        """
        prepro_states = []
        prepro_dists = []
        for state, dist in zip(states, dists):
            nan = state == dat.CPG_NAN
            if np.any(nan):
                # Set CpG neighbors at the flanks of a chromosome to 0.5
                state[nan] = 0.5
                dist[nan] = self.cpg_max_dist
            dist = np.minimum(dist, self.cpg_max_dist) / self.cpg_max_dist
            prepro_states.append(np.expand_dims(state, 1))
            prepro_dists.append(np.expand_dims(dist, 1))
        prepro_states = np.concatenate(prepro_states, axis=1)
        prepro_dists = np.concatenate(prepro_dists, axis=1)
        if self.cpg_wlen:
            center = prepro_states.shape[2] // 2
            delta = self.cpg_wlen // 2
            tmp = slice(center - delta, center + delta)
            prepro_states = prepro_states[:, :, tmp]
            prepro_dists = prepro_dists[:, :, tmp]
        return (prepro_states, prepro_dists)

    @dat.threadsafe_generator
    def __call__(self, data_files, class_weights=None, *args, **kwargs):
        """Return generator for reading data from `data_files`.

        Parameters
        ----------
        data_files: list
            List of data files to be read.
        class_weights: dict
            dict of dict with class weights of individual outputs.
        *args: list
            Unnamed arguments passed to :func:`hdf.reader`
        *kwargs: dict
            Named arguments passed to :func:`hdf.reader`

        Returns
        -------
        generator
            Python generator for reading data.
        """
        names = []
        if self.use_dna:
            names.append('inputs/dna')

        if self.replicate_names:
            for name in self.replicate_names:
                names.append('inputs/cpg/%s/state' % name)
                names.append('inputs/cpg/%s/dist' % name)

        if self.output_names:
            for name in self.output_names:
                names.append('outputs/%s' % name)

        for data_raw in hdf.reader(data_files, names, *args, **kwargs):
            inputs = dict()

            if self.use_dna:
                inputs['dna'] = self._prepro_dna(data_raw['inputs/dna'])

            if self.replicate_names:
                states = []
                dists = []
                for name in self.replicate_names:
                    tmp = 'inputs/cpg/%s/' % name
                    states.append(data_raw[tmp + 'state'])
                    dists.append(data_raw[tmp + 'dist'])
                states, dists = self._prepro_cpg(states, dists)
                if self.encode_replicates:
                    # DEPRECATED: to support loading data for legacy models
                    tmp = '/' + encode_replicate_names(self.replicate_names)
                else:
                    tmp = ''
                inputs['cpg/state%s' % tmp] = states
                inputs['cpg/dist%s' % tmp] = dists

            if not self.output_names:
                yield inputs
            else:
                outputs = dict()
                weights = dict()

                for name in self.output_names:
                    outputs[name] = data_raw['outputs/%s' % name]
                    cweights = class_weights[name] if class_weights else None
                    weights[name] = get_sample_weights(outputs[name], cweights)
                    if name.endswith('cat_var'):
                        output = outputs[name]
                        outputs[name] = to_categorical(output, 3)
                        outputs[name][output == dat.CPG_NAN] = 0

                yield (inputs, outputs, weights)


def data_reader_from_model(model, outputs=True, replicate_names=None):
    """Return :class:`DataReader` from `model`.

    Builds a :class:`DataReader` for reading data for `model`.

    Parameters
    ----------
    model: :class:`Model`.
        :class:`Model`.
    outputs: bool
        If `True`, return output labels.
    replicate_names: list
        Name of input cells of `model`.

    Returns
    -------
    :class:`DataReader`
        Instance of :class:`DataReader`.
    """
    use_dna = False
    dna_wlen = None
    cpg_wlen = None
    output_names = None
    encode_replicates = False

    input_shapes = to_list(model.input_shape)
    for input_name, input_shape in zip(model.input_names, input_shapes):
        if input_name == 'dna':
            # Read DNA sequences.
            use_dna = True
            dna_wlen = input_shape[1]
        elif input_name.startswith('cpg/state/'):
            # DEPRECATED: legacy model. Decode replicate names from input name.
            replicate_names = decode_replicate_names(
                input_name.replace('cpg/state/', ''))
            assert len(replicate_names) == input_shape[1]
            cpg_wlen = input_shape[2]
            encode_replicates = True
        elif input_name == 'cpg/state':
            # Read neighboring CpG sites.
            if not replicate_names:
                raise ValueError('Replicate names required!')
            if len(replicate_names) != input_shape[1]:
                tmp = '{r} replicates found but CpG model was trained with' \
                    ' {s} replicates. Use `--nb_replicate {s}` or ' \
                    ' `--replicate_names` option to select {s} replicates!'
                tmp = tmp.format(r=len(replicate_names), s=input_shape[1])
                raise ValueError(tmp)
            cpg_wlen = input_shape[2]

    if outputs:
        # Return output labels.
        output_names = model.output_names

    return DataReader(output_names=output_names,
                      use_dna=use_dna,
                      dna_wlen=dna_wlen,
                      cpg_wlen=cpg_wlen,
                      replicate_names=replicate_names,
                      encode_replicates=encode_replicates)
