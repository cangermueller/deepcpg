from os import path as pt

import h5py as h5
import numpy as np

from keras import layers as kl
from keras import models as km
from keras import regularizers as kr

from ..data.preprocess import CPG_NAN
from ..data import dna
from ..data import io

"""TODO
Residual block
Dilated convs
Strided conv instead of max pooling
Global average pooling
Try batch norm inputs
CpG as before or separate networks?
"""


def data_reader(data_files, outputs,
                dna_wlen=None, cpg_wlen=None, cpg_max_dist=25000,
                class_weights=None, *args, **kwargs):
    """
    dna_wlen=None: no dna
    dna_wlen=0: all
    """

    names = {'outputs': outputs}

    if dna_wlen is not None:
        names.append('inputs/dna')

    if cpg_wlen is not None:
        for output in outputs:
            names.append('inputs/cpg_context/%s/state' % output)
            names.append('inputs/cpg_context/%s/dist' % output)

    for data_raw in io.h5_reader(data_files, names, *args, **kwargs):
        inputs = dict()
        outputs = dict()
        weights = dict()

        if dna_wlen is not None:
            dna = data_raw['inputs/dna']
            if dna_wlen > 0:
                cur_wlen = dna.shape[1]
                center = cur_wlen // 2
                delta = dna_wlen // 2
                dna = dna[:, center - delta, center + delta + 1]
            dna = dna.int2onehot(dna)
            inputs['inputs/dna'] = dna

        if cpg_wlen is not None:
            for output in outputs:
                state = data_raw['inputs/cpg_context/%s/state' % output]
                dist = data_raw['inputs/cpg_context/%s/dist' % output]
                nan = state == CPG_NAN
                if np.any(nan):
                    tmp = np.sum(state == 1) / state.size
                    state[nan] = np.random.binomial(1, tmp, nan.sum())
                    dist[nan] = cpg_max_dist
                dist = np.minimum(dist, cpg_max_dist) / cpg_max_dist
                cpg = np.empty(list(state.shape) + [2], dtype=np.float32)
                cpg[:, :, 0] = state
                cpg[:, :, 1] = dist
                if cpg_wlen > 0:
                    cur_wlen = cpg.shape[1]
                    center = cur_wlen // 2
                    delta = cpg_wlen // 2
                    cpg = cpg[:, center - delta, center + delta]
                inputs['inputs/cpg_context/%s' % output] = cpg

        for output in outputs:
            name = 'outputs/%s' % output
            inputs[name] = data_raw[name]
            cweights = class_weights[name] if class_weights else None
            weights[name] = get_sample_weights(inputs[name], cweights)

        yield (inputs, outputs, weights)


def data_generator(data_files, targets, batch_size=128, nb_sample=None,
                   dna_wlen=None, cpg_wlen=None, cpg_max_dist=25000,
                   class_weights=None, shuffle=True,
                   loop=True):
    """
    dna_wlen=None: no dna
    dna_wlen=0: all
    """

    file_idx = 0
    nb_seen = 0
    data_files = list(data_files)
    if nb_sample is None:
        nb_sample = np.inf

    while True:
        if shuffle and file_idx == 0:
            np.random.shuffle(data_files)
        data_file = h5.File(data_files[file_idx], 'r')
        nb_sample_file = len(data_file['pos'])
        nb_batch = nb_sample_file // batch_size

        if dna_wlen is not None:
            cur_wlen = data_file['dna'].shape[1]
            if dna_wlen > 0:
                center = cur_wlen // 2
                delta = dna_wlen // 2
                dna_cols = slice(center - delta, center + delta + 1)
            else:
                dna_cols = slice(0, cur_wlen)

        if cpg_wlen is not None:
            cur_wlen = data_file['cpg_context'][targets[0]]['state']
            cur_wlen = cur_wlen.shape[1]
            if cpg_wlen > 0:
                center = cur_wlen // 2
                delta = cpg_wlen // 2
                cpg_cols = slice(center - delta, center + delta)
            else:
                cpg_cols = slice(0, cur_wlen)

        for batch in range(nb_batch):
            batch_start = batch * batch_size
            batch_end = min(nb_sample_file, batch_start + batch_size)
            nb_seen += batch_end - batch_start
            if nb_seen > nb_sample:
                data_files = data_files[:file_idx + 1]
                break

            xs = dict()
            if dna_wlen is not None:
                x = data_file['dna'][batch_start:batch_end, dna_cols]
                x = dna.int2onehot(x)
                xs['x/dna'] = x

            if cpg_wlen is not None:
                for target in targets:
                    group = data_file['cpg_context'][target]
                    state = group['state'][batch_start:batch_end, cpg_cols]
                    dist = group['dist'][batch_start:batch_end, cpg_cols]
                    nan = state == CPG_NAN
                    # TODO: Improve normalization and random 0.5
                    if np.any(nan):
                        tmp = np.sum(state == 1) / state.size
                        state[nan] = np.random.binomial(1, tmp, nan.sum())
                        dist[nan] = cpg_max_dist
                    dist = np.minimum(dist, cpg_max_dist) / cpg_max_dist
                    x = np.empty(list(state.shape) + [2], dtype=np.float32)
                    x[:, :, 0] = state
                    x[:, :, 1] = dist
                    xs['x/cpg'] = x

            ys = dict()
            ws = dict()
            for target in targets:
                y = data_file['cpg'][target][batch_start:batch_end]
                name = 'cpg/%s' % target
                ws[name] = get_sample_weights(y, class_weights)
                ys[name] = y

            yield (xs, ys, ws)

        data_file.close()
        file_idx += 1
        if file_idx >= len(data_files):
            file_idx = 0
            nb_seen = 0


def model01(x, dropout=0.0, l1_decay=0.0, l2_decay=0.0):
    w_reg = kr.WeightRegularizer(l1=l1_decay, l2=l2_decay)
    bn_axis = 2

    x = kl.Conv1D(64, 9, init='he_uniform', W_regularizer=w_reg)(x)
    x = kl.BatchNormalization(axis=bn_axis)(x)
    x = kl.Activation('relu')(x)
    x = kl.Dropout(dropout)(x)
    x = kl.MaxPooling1D(2, 2)(x)

    x = kl.Conv1D(128, 3, init='he_uniform', W_regularizer=w_reg)(x)
    x = kl.BatchNormalization(axis=bn_axis)(x)
    x = kl.Activation('relu')(x)
    x = kl.Dropout(dropout)(x)
    x = kl.MaxPooling1D(2, 2)(x)

    x = kl.Conv1D(256, 3, init='he_uniform', W_regularizer=w_reg)(x)
    x = kl.BatchNormalization(axis=bn_axis)(x)
    x = kl.Activation('relu')(x)
    x = kl.Dropout(dropout)(x)
    x = kl.MaxPooling1D(2, 2)(x)

    x = kl.Conv1D(512, 3, init='he_uniform', W_regularizer=w_reg)(x)
    x = kl.BatchNormalization(axis=bn_axis)(x)
    x = kl.Activation('relu')(x)
    x = kl.Dropout(dropout)(x)
    x = kl.MaxPooling1D(2, 2)(x)

    x = kl.GlobalAveragePooling1D()(x)

    return x
