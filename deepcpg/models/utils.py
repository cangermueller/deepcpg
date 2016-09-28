from os import path as pt

from keras import models as km
from keras import layers as kl
import numpy as np


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


def add_outputs(x, targets):
    outputs = []
    for target in targets:
        output = kl.Dense(1, init='he_uniform', activation='sigmoid',
                          name='cpg/%s' % target)(x)
        outputs.append(output)
    return outputs
