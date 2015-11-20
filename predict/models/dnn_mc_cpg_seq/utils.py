import h5py as h5
import pandas as pd
import numpy as np
import random

from predict.evaluation import evaluate

MASK = -1


def load_model(json_file, weights_file=None):
    import keras.models as kmodels
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


class DataReader(object):

    def __init__(self, path, chromos=None, shuffle=True, chunk_size=1, loop=False):
        self.path = path
        if chromos is None:
            f = h5.File(self.path)
            chromos = sorted([x for x in f.keys() if x.isdigit()])
            f.close()
        self.chromos = chromos
        self.shuffle = shuffle
        self.chunk_size = chunk_size
        self.loop = loop

    def __iter__(self):
        self._iter_chromos = list(reversed(self.chromos))
        if self.shuffle:
            random.shuffle(self._iter_chromos)
        self._iter_idx = []
        return self

    def __next__(self):
        if len(self._iter_idx) == 0:
            if len(self._iter_chromos) == 0:
                if self.loop:
                    iter(self)
                else:
                    raise StopIteration
            self._iter_chromo = self._iter_chromos.pop()
            f = h5.File(self.path)
            n = f['/%s/pos' % (self._iter_chromo)].shape[0]
            f.close()
            self._iter_idx = list(reversed(range(0, n, self.chunk_size)))
            if self.shuffle:
                random.shuffle(self._iter_idx)
        self._iter_i = self._iter_idx.pop()
        self._iter_j = self._iter_i + self.chunk_size
        return (self._iter_chromo, self._iter_i, self._iter_j)

    def next(self):
        return self.__next__()
