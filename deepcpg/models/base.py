import h5py as h5
import numpy as np

from keras import layers as kl
from keras import regularizers as kr

from ..data import dna

"""TODO
Residual block
Dilated convs
Global average pooling
"""


def get_sample_weights(y, class_weights):
    y = y[:]
    if not class_weights:
        class_weights = {0: 0.5, 1: 0.5}
    sample_weights = np.zeros(y.shape, dtype='float16')
    for cla, weight in class_weights.items():
        sample_weights[y == cla] = weight
    return sample_weights


def get_class_weights(data_files, target):
    nb_zero = 0
    nb_sample = 0
    for data_file in data_files:
        data_file = h5.File(data_file, 'r')
        y = data_file['cpg'][target].value
        nb_zero += y == 0
        nb_sample += len(y)
    frac_zero = nb_zero / nb_sample
    return {0: 1 - frac_zero, 1: frac_zero}


def save_model(model, json_file, weights_file=None):
    with open(json_file, 'w') as f:
        f.write(model.to_json())
    if weights_file is not None:
        model.save_weights(weights_file, overwrite=True)


def load_model(json_file, weights_file=None):
    from keras import models as kmod

    with open(json_file, 'r') as f:
        model = f.read()
    model = kmod.model_from_json(model)
    if weights_file:
        model.load_weights(weights_file)


# TODO: Allow to use smaller wlen
# TODO: One-hot encode
def data_generator(data_files, targets, batch_size=128, nb_sample=None,
                   target_filter=None, class_weights=None, shuffle=True,
                   loop=True):
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
        for batch in range(nb_batch):
            batch_start = batch * batch_size
            batch_end = min(nb_sample_file, batch_start + batch_size)
            nb_seen += batch_end - batch_start
            if nb_seen > nb_sample:
                data_files = data_files[:file_idx]
                break

            xs = []
            if 'dna' in data_file:
                x = dna.int2onehot(data_file['dna'][batch_start:batch_end])
                xs.append(x)

            ys = []
            ws = []
            for target in targets:
                y = data_file['cpg'][target][batch_start:batch_end]
                ws.append(get_sample_weights(y, class_weights))
                ys.append(y)

            yield (xs, ys, ws)

        data_file.close()
        file_idx += 1
        if file_idx >= len(data_files):
            file_idx = 0
            nb_seen = 0


def model(x, dropout=0.0, l1_decay=0.0, l2_decay=0.0):
    w_reg = kr.WeightRegularizer(l1=l1_decay, l2=l2_decay)
    x = kl.Conv1D(64, 9, init='he_uniform', W_regularizer=w_reg)(x)
    x = kl.BatchNormalization()(x)
    x = kl.Activation('relu')(x)
    x = kl.Dropout(dropout)(x)
    x = kl.MaxPooling1D(2, 2)(x)

    x = kl.Conv1D(128, 3, init='he_uniform', W_regularizer=w_reg)(x)
    x = kl.BatchNormalization()(x)
    x = kl.Activation('relu')(x)
    x = kl.Dropout(dropout)(x)
    x = kl.MaxPooling1D(2, 2)(x)

    x = kl.Conv1D(256, 3, init='he_uniform', W_regularizer=w_reg)(x)
    x = kl.BatchNormalization()(x)
    x = kl.Activation('relu')(x)
    x = kl.Dropout(dropout)(x)
    x = kl.MaxPooling1D(2, 2)(x)

    x = kl.Conv1D(512, 3, init='he_uniform', W_regularizer=w_reg)(x)
    x = kl.BatchNormalization()(x)
    x = kl.Activation('relu')(x)
    x = kl.Dropout(dropout)(x)
    x = kl.MaxPooling1D(2, 2)(x)

    x = kl.GlobalAveragePooling1D()(x)

    return x


def add_cpg_outputs(x, targets):
    outputs = []
    for target in targets:
        output = kl.Dense(1, init='he_uniform', activation='sigmoid',
                          name='cpg/%s' % target)(x)
        outputs.append(output)
    return outputs
