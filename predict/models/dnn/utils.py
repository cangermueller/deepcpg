import h5py as h5
import numpy as np
import re

import predict.models.dnn.callbacks as cbk

MASK = -1


def make_cbk():
    return cbk.EarlyStopping()


def read_targets(path, targets=None):
    f = h5.File(path)
    g = f['targets']
    tar = dict()
    for k in g.keys():
        tar[k] = [x.decode() for x in g[k].value]
    f.close()
    if targets is not None:
        idx = []
        for i, x in enumerate(tar['name']):
            for target in targets:
                if re.search(target, x):
                    idx.append(i)
        for k, v in tar.items():
            tar[k] = [v[i] for i in idx]
    return tar


def target_id2name(ids, targets):
    targets_map = {x[0]: x[1] for x in zip(targets['id'], targets['name'])}
    names = [targets_map[x.replace('_y', '')] for x in ids]
    return names


def write_z(data, z, targets, out_file, unlabeled=False, name='z',
            overwrite=True):
    target_map = dict()
    for x in zip(targets['id'], targets['name']):
        target_map[x[0] + '_y'] = x[1]

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
                if k in gtc and (k == name or overwrite):
                    del gtc[k]
                if k not in gtc and k != 'chromo':
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


def read_data(path, max_mem=None):
    f = open_hdf(path, cache_size=max_mem)
    data = dict()
    for k, v in f['data'].items():
        data[k] = v
    for k, v in f['pos'].items():
        data[k] = v
    return (f, data)


def select_data(data, chromo, start=None, end=None, log=None):
    sel = data['chromo'].value == str(chromo).encode()
    if start is not None:
        sel &= data['pos'].value >= start
    if end is not None:
        sel &= data['pos'].value <= end
    if sel.sum() == 0:
        return 0
    if log is not None:
        log.info('Select %d samples' % (sel.sum()))
    for k in data.keys():
        if len(data[k].shape) > 1:
            data[k] = data[k][sel, :]
        else:
            data[k] = data[k][sel]
    return sel.sum()


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

    def _adapt_slice(self, key):
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
        return idx

    def _adapt_key(self, key):
        if isinstance(key, slice):
            idx = self._adapt_slice(key)
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
        return idx

    def __getitem__(self, key):
        if isinstance(key, tuple):
            idx = list(key)
            idx[0] = self._adapt_key(idx[0])
            idx = tuple(idx)
        else:
            idx = self._adapt_key(key)
        return self.data[idx]

    def use_all(self):
        self.start = 0
        self.stop = self.data.shape[0]

    @property
    def shape(self):
        return tuple([len(self)] + list(self.data.shape[1:]))


def read_hdf(path, cache_size):
    f = open_hdf(path, cache_size=cache_size)
    data = dict()
    for k, v in f['data'].items():
        data[k] = v
    for k, v in f['pos'].items():
        data[k] = v
    return (f, data)


def to_view(d, *args, **kwargs):
    for k in d.keys():
        d[k] = ArrayView(d[k], *args, **kwargs)


def select_cpos(data, chromo, start=None, end=None):
    sel = data['chromo'].value == str(chromo).encode()
    if start is not None:
        sel &= data['pos'].value >= start
    if end is not None:
        sel &= data['pos'].value <= end
    for k in data.keys():
        if len(data[k].shape) > 1:
            data[k] = data[k][sel, :]
        else:
            data[k] = data[k][sel]
    return data
