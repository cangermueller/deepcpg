import h5py as h5
import keras.models as kmodels
import pandas as pd
import numpy as np

from predict.evaluation import evaluate

MASK = -1


def load_model(json_file, weights_file=None):
    with open(json_file, 'r') as f:
        model = f.read()
    model = kmodels.model_from_json(model)
    model.load_weights(weights_file)
    return model

def read_data(path, max_samples=None):
    f = h5.File(path, 'r')
    data = dict()
    for k, v in f.items():
        if max_samples is None or k.startswith('label'):
            data[k] = v.value
        else:
            data[k] = v[:max_samples]
    f.close()
    data['c_x'] = data['c_x'].astype('float32')
    dist_max = np.iinfo('int32').max - 100
    data['c_x'][:, 1] /= dist_max
    return data

def evaluate_all(y, z):
    keys = sorted(z.keys())
    p = [evaluate(y[k], z[k]) for k in keys]
    p = pd.concat(p)
    p.index = keys
    return p
