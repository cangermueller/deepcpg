import os

from keras import backend as K
from keras import models as km
from keras import layers as kl
from keras.utils.np_utils import to_categorical
import numpy as np
import pandas as pd

from .. import data as dat
from .. import evaluation as ev
from ..data import hdf
from ..data.dna import int_to_onehot
from ..utils import to_list


class ScaledSigmoid(kl.Layer):

    def __init__(self, scaling, **kwargs):
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


def get_sample_weights(y, class_weights=None):
    y = y[:]
    sample_weights = np.ones(y.shape, dtype=K.floatx())
    sample_weights[y == dat.CPG_NAN] = K.epsilon()
    if class_weights is not None:
        for cla, weight in class_weights.items():
            sample_weights[y == cla] = weight
    return sample_weights


def save_model(model, model_file, weights_file=None):
    if os.path.splitext(model_file)[1] == '.h5':
        model.save(model_file)
    else:
        with open(model_file, 'w') as f:
            f.write(model.to_json())
    if weights_file is not None:
        model.save_weights(weights_file, overwrite=True)


def load_model(model_files, custom_objects=CUSTOM_OBJECTS):
    if not isinstance(model_files, list):
        model_files = [model_files]
    if os.path.splitext(model_files[0])[1] == '.h5':
        model = km.load_model(model_files[0], custom_objects=custom_objects)
    else:
        with open(model_files[0], 'r') as f:
            model = f.read()
        model = km.model_from_json(model, custom_objects=custom_objects)
    if len(model_files) > 1:
        model.load_weights(model_files[1])
    return model


def get_objectives(output_names):
    objectives = dict()
    for output_name in output_names:
        if output_name.startswith('cpg'):
            objective = 'binary_crossentropy'
        elif output_name.startswith('bulk'):
            objective = 'mean_squared_error'
        elif output_name in ['stats/diff', 'stats/mode', 'stats/cat2_var']:
            objective = 'binary_crossentropy'
        elif output_name in ['stats/mean', 'stats/var']:
            objective = 'mean_squared_error'
        elif output_name in ['stats/cat_var']:
            objective = 'categorical_crossentropy'
        else:
            raise ValueError('Invalid output name "%s"!')
        objectives[output_name] = objective
    return objectives


def add_output_layers(stem, output_names):
    outputs = []
    for output_name in output_names:
        if output_name == 'stats/var':
            x = kl.Dense(1, init='he_uniform')(stem)
            x = ScaledSigmoid(0.25, name=output_name)(x)
        elif output_name == 'stats/cat_var':
            x = kl.Dense(3, init='he_uniform',
                         activation='softmax',
                         name=output_name)(stem)
        else:
            x = kl.Dense(1, init='he_uniform',
                         activation='sigmoid',
                         name=output_name)(stem)
        outputs.append(x)
    return outputs


def predict_generator(model, generator, nb_sample=None):
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


def evaluate_generator(model, generator, nb_sample, return_data=False):
    data = predict_generator(model, generator, nb_sample)
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


class Model(object):

    def __init__(self, dropout=0.0, l1_decay=0.0, l2_decay=0.0,
                 init='he_uniform'):
        self.dropout = dropout
        self.l1_decay = l1_decay
        self.l2_decay = l2_decay
        self.init = init
        self.name = self.__class__.__name__

    def inputs(self, *args, **kwargs):
        pass

    def __call__(self, inputs=None):
        pass


def encode_replicate_names(replicate_names):
    return '--'.join(replicate_names)


def decode_replicate_names(replicate_names):
    return replicate_names.split('--')


class DataReader(object):

    def __init__(self, output_names=None,
                 use_dna=True, dna_wlen=None,
                 replicate_names=None, cpg_wlen=None, cpg_max_dist=25000):
        self.output_names = to_list(output_names)
        self.use_dna = use_dna
        self.dna_wlen = dna_wlen
        self.replicate_names = to_list(replicate_names)
        self.cpg_wlen = cpg_wlen
        self.cpg_max_dist = cpg_max_dist

    def _prepro_dna(self, dna):
        if self.dna_wlen:
            cur_wlen = dna.shape[1]
            center = cur_wlen // 2
            delta = self.dna_wlen // 2
            dna = dna[:, (center - delta):(center + delta + 1)]
        return int_to_onehot(dna)

    def _prepro_cpg(self, states, dists):
        prepro_states = []
        prepro_dists = []
        for state, dist in zip(states, dists):
            nan = state == dat.CPG_NAN
            if np.any(nan):
                tmp = np.sum(state == 1) / state.size
                state[nan] = np.random.binomial(1, tmp, nan.sum())
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
                replicates_id = encode_replicate_names(self.replicate_names)
                inputs['cpg/state/%s' % replicates_id] = states
                inputs['cpg/dist/%s' % replicates_id] = dists

            if not self.output_names:
                yield inputs
            else:
                outputs = dict()
                weights = dict()

                for name in self.output_names:
                    outputs[name] = data_raw['outputs/%s' % name]
                    cweights = class_weights[name] if class_weights else None
                    weights[name] = get_sample_weights(outputs[name], cweights)
                    if name == 'stats/cat_var':
                        output = outputs[name]
                        outputs[name] = to_categorical(output, 3)
                        outputs[name][output == dat.CPG_NAN] = 0

                yield (inputs, outputs, weights)


def data_reader_from_model(model):
    use_dna = False
    dna_wlen = None
    cpg_wlen = None
    replicate_names = None

    input_shapes = to_list(model.input_shape)
    for input_name, input_shape in zip(model.input_names, input_shapes):
        if input_name == 'dna':
            use_dna = True
            dna_wlen = input_shape[1]
        elif input_name.startswith('cpg/state/'):
            replicate_names = decode_replicate_names(
                input_name.replace('cpg/state/', ''))
            assert len(replicate_names) == input_shape[1]
            cpg_wlen = input_shape[2]

    return DataReader(output_names=model.output_names,
                      use_dna=use_dna,
                      dna_wlen=dna_wlen,
                      cpg_wlen=cpg_wlen,
                      replicate_names=replicate_names)
