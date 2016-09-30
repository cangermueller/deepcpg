from os import path as pt

from keras import models as km
from keras import layers as kl
import numpy as np

from ..data import CPG_NAN, h5_reader
from ..data.dna import int2onehot


def get_sample_weights(y, class_weights):
    y = y[:]
    if not class_weights:
        class_weights = {0: 0.5, 1: 0.5}
    sample_weights = np.zeros(y.shape, dtype='float16')
    for cla, weight in class_weights.items():
        sample_weights[y == cla] = weight
    return sample_weights


def save_model(model, model_file, weights_file=None):
    if pt.splitext(model_file)[1] == '.h5':
        model.save(model_file)
    else:
        with open(model_file, 'w') as f:
            f.write(model.to_json())
    if weights_file is not None:
        model.save_weights(weights_file, overwrite=True)


def load_model(model_files):
    if not isinstance(model_files, list):
        model_files = [model_files]
    if pt.splitext(model_files[0])[1] == '.h5':
        model = km.load_model(model_files[0])
    else:
        with open(model_files[0], 'r') as f:
            model = f.read()
        model = km.model_from_json(model)
    if len(model_files) > 1:
        model.load_weights(model_files[1])
    return model


def add_outputs(x, output_names):
    outputs = []
    for name in output_names:
        output = kl.Dense(1, init='he_uniform', activation='sigmoid',
                          name='%s' % name)(x)
        outputs.append(output)
    return outputs


class Model(object):

    def __init__(self, dropout=0.0, l1_decay=0.0, l2_decay=0.0):
        self.dropout = dropout
        self.l1_decay = l1_decay
        self.l2_decay = l2_decay

    def _prepro_dna(self, dna, dna_wlen=None):
        if dna_wlen:
            cur_wlen = dna.shape[1]
            center = cur_wlen // 2
            delta = dna_wlen // 2
            dna = dna[:, (center - delta):(center + delta + 1)]
        return int2onehot(dna)

    def _prepro_cpg(self, states, dists, cpg_max_dist):
        prepro_states = []
        prepro_dists = []
        for state, dist in zip(states, dists):
            nan = state == CPG_NAN
            if np.any(nan):
                tmp = np.sum(state == 1) / state.size
                state[nan] = np.random.binomial(1, tmp, nan.sum())
                dist[nan] = cpg_max_dist
            dist = np.minimum(dist, cpg_max_dist) / cpg_max_dist
            prepro_states.append(np.expand_dims(state, 1))
            prepro_dists.append(np.expand_dims(dist, 1))
        prepro_states = np.concatenate(prepro_states, axis=1)
        prepro_dists = np.concatenate(prepro_dists, axis=1)
        return (prepro_states, prepro_dists)

    def reader(self, data_files, output_names=None,
               use_dna=True, dna_wlen=None,
               cpg_names=None, cpg_wlen=None, cpg_max_dist=25000,
               class_weights=None, *args, **kwargs):

        names = []
        if use_dna:
            names.append('inputs/dna')

        if cpg_names:
            for name in cpg_names:
                names.append('inputs/cpg_context/%s/state' % name)
                names.append('inputs/cpg_context/%s/dist' % name)

        if output_names:
            names.extend(['outputs/%s' % name for name in output_names])

        for data_raw in h5_reader(data_files, names, *args, **kwargs):
            inputs = dict()

            if use_dna:
                inputs['dna'] = self._prepro_dna(data_raw['inputs/dna'],
                                                 dna_wlen)

            if cpg_names:
                states = []
                dists = []
                for cpg_name in cpg_names:
                    tmp = 'inputs/cpg_context/%s/' % cpg_name
                    states.append(data_raw[tmp + 'state'])
                    dists.append(data_raw[tmp + 'dist'])
                states, dists = self._prepro_cpg(states, dists, cpg_wlen)
                inputs['cpg_context/state'] = states
                inputs['cpg_context/dist'] = dists

            if not output_names:
                yield inputs
            else:
                outputs = dict()
                weights = dict()

                for name in output_names:
                    outputs[name] = data_raw['outputs/%s' % name]
                    cweights = class_weights[name] if class_weights else None
                    weights[name] = get_sample_weights(outputs[name], cweights)

                yield (inputs, outputs, weights)

    def inputs(self, *args, **kwargs):
        pass

    def __call__(self, inputs):
        pass
