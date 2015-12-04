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


def evaluate_all(y, z):
    keys = sorted(z.keys())
    p = [evaluate(y[k][:], z[k][:]) for k in keys]
    p = pd.concat(p)
    p.index = keys
    return p


def read_and_stack(reader, callback):
    x = []
    for o in reader:
        x.append(callback(*o))
    y = dict()
    for k in x[0].keys():
        y[k] = dict()
        for l in x[0][k].keys():
            y[k][l] = np.hstack([x[i][k][l] for i in range(len(x))])
    return y


def read_labels(path):
    f = h5.File(path)
    g = f['labels']
    l = dict()
    for k in g.keys():
        l[k] = [x.decode() for x in g[k].value]
    f.close()
    return l


def map_targets(targets, labels):
    targets = [x.replace('_y', '') for x in targets]
    targets_map = {x[0]: x[1] for x in zip(labels['targets'], labels['files'])}
    targets = [targets_map[x] for x in targets]
    return targets


def write_z(data, z, labels, out_file):
    target_map = {x[0] + '_y': x[1] for x in zip(labels['targets'], labels['files'])}

    f = h5.File(out_file, 'w')
    for target in z.keys():
        d = dict()
        d['z'] = np.ravel(z[target])
        d['y'] = np.ravel(data[target][:])
        d['pos'] = data['pos'][:]
        d['chromo'] = data['chromo'][:]
        t = d['y'] != MASK
        for k in d.keys():
            d[k] = d[k][t]

        gt = f.create_group(target_map[target])
        for chromo in np.unique(d['chromo']):
            t = d['chromo'] == chromo
            dc = {k: d[k][t] for k in d.keys()}
            t = np.argsort(dc['pos'])
            for k in dc.keys():
                dc[k] = dc[k][t]
            t = dc['pos']
            assert np.all(t[:-1] < t[1:])

            gtc = gt.create_group(chromo)
            for k in dc.keys():
                gtc[k] = dc[k]
    f.close()


def write_z2(data, z, labels, out_file, unlabeled=False, name='z'):
    target_map = {x[0] + '_y': x[1] for x in zip(labels['targets'], labels['files'])}

    f = h5.File(out_file, 'a')
    for target in z.keys():
        d = dict()
        d[name] = np.ravel(z[target])
        d['y'] = np.ravel(data[target][:])
        d['pos'] = data['pos'][:]
        d['chromo'] = data['chromo'][:]
        if not unlabeled:
            t = d['y'] != MASK
            for k in d.keys():
                d[k] = d[k][t]

        t = target_map[target]
        if t in f:
            gt = f[t]
        else:
            gt = f.create_group(t)
        for chromo in np.unique(d['chromo']):
            t = d['chromo'] == chromo
            dc = {k: d[k][t] for k in d.keys()}
            t = np.argsort(dc['pos'])
            for k in dc.keys():
                dc[k] = dc[k][t]
            t = dc['pos']
            assert np.all(t[:-1] < t[1:])

            if chromo in gt:
                gtc = gt[chromo]
            else:
                gtc = gt.create_group(chromo)
            for k in dc.keys():
                if k not in gtc:
                    gtc[k] = dc[k]
    f.close()






def open_hdf(filename, acc='r', cache_size=None):
    if cache_size:
        propfaid = h5.h5p.create(h5.h5p.FILE_ACCESS)
        settings = list(propfaid.get_cache())
        settings[2] = cache_size
        propfaid.set_cache(*settings)
        fid = h5.h5f.open(filename.encode(), fapl=propfaid)
        _file = h5.File(fid, acc)
    else:
        _file = h5.File(filename, acc)
    return _file


class DataReader(object):

    def __init__(self, path, chromos=None, shuffle=False, chunk_size=1,
                 loop=False, max_chunks=None):
        self.path = path
        if chromos is None:
            chromos = read_chromos(self.path)
        self.chromos = chromos
        self.shuffle = shuffle
        self.chunk_size = chunk_size
        self.loop = loop
        self.max_chunks = max_chunks

    def __iter__(self):
        self._iter_chromos = list(reversed(self.chromos))
        if self.shuffle:
            random.shuffle(self._iter_chromos)
        self._iter_idx = []
        self._n = 0
        return self

    def __next__(self):
        if self.max_chunks is not None and self._n == self.max_chunks:
            raise StopIteration
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
        self._n += 1
        return (self._iter_chromo, self._iter_i, self._iter_j)

    def next(self):
        return self.__next__()


class ArrayView(object):

    def __init__(self, data, start=0, stop=None):
        self.start = start
        if stop is None:
            stop = data.shape[0]
        else:
            stop = min(stop, data.shape[0])
        self.stop = stop
        self.data = data

    def __len__(self):
        return self.stop - self.start

    def __getitem__(self, key):
        if isinstance(key, slice):
            start = key.start
            if start is None:
                start = 0
            stop = key.stop
            if stop is None:
                stop = self.stop - self.start
            if self.start + stop <= self.stop:
                idx = slice(self.start + start,
                            self.start + stop)
            else:
                raise IndexError
        elif isinstance(key, int):
            if self.start + key < self.stop:
                idx = self.start + key
            else:
                raise IndexError
        elif isinstance(key, np.ndarray):
            if self.start + np.max(key) < self.stop:
                idx = (self.start + key).tolist()
            else:
                raise IndexError
        elif isinstance(key, list):
            if self.start + max(key) < self.stop:
                idx = [self.start + x for x in key]
            else:
                raise IndexError
        return self.data[idx]

    @property
    def shape(self):
        return tuple([len(self)] + list(self.data.shape[1:]))
