import os.path as pt
import pandas as pd
import numpy as np
import re

from predict import data, hdf


class FeatureSelection(object):

    def __init__(self):
        self.cpg = False  # True, False, None
        self.knn = False  # k, True, False, None
        self.knn_dist = False  # k, True, False, None
        self.annos = False  # non-empty list, True, False, None
        self.annos_dist = False  # non-empty list, True, False, None
        self.scores = False  # non-empty list, True, False, None


class RangeSelection(object):

    def __init__(self, chromo='1', start=None, end=None):
        self.chromo = chromo
        self.start = start
        self.end = end

    def query(self):
        s = []
        if self.start:
            s.append('start >= %d' % self.start)
        if self.end:
            s.append('end >= %d' % self.end)
        if len(s):
            return ' & '.join(s)
        else:
            return None


def select_cpg(path, dataset, range_sel, samples=None):
    g = pt.join(dataset, 'cpg', range_sel.chromo)
    if samples is None:
        samples = hdf.ls(path, g)
    d = []
    for sample in samples:
        gs = pt.join(g, sample)
        f = pd.read_hdf(path, gs, where=range_sel.query())
        f['feature'] = sample
        d.append(f)
    d = pd.concat(d)
    return d


def select_knn(path, dataset, range_sel, samples=None, k=None, dist=False):
    if k is None or type(k) is bool:
        p = r'knn\d'
        if dist:
            p += '_dist'
        p += '$'
        t = hdf.ls(path, dataset)
        t = [x for x in t if re.match(p, x)]
        name = t[0]
    else:
        name = 'knn%d' % (k)
        if dist:
            name += '_dist'

    g = pt.join(dataset, name, range_sel.chromo)
    if samples is None:
        samples = hdf.ls(path, g)
    d = []
    for sample in samples:
        gs = pt.join(g, sample)
        f = pd.read_hdf(path, gs, where=range_sel.query())
        if dist:
            t = 'dist_'
        else:
            t = 'cpg_'
        f['feature'] = [sample + '_' + x.replace(t, '') for x in f.feature]
        d.append(f)
    d = pd.concat(d)
    return d


def select_annos(path, dataset, range_sel, annos=None, dist=False):
    group = 'annos'
    if dist:
        group += '_dist'
    if annos is None or type(annos) is bool:
        annos = hdf.ls(path, pt.join(dataset, group))
    d = []
    for anno in annos:
        gs = pt.join(dataset, group, anno, range_sel.chromo)
        f = pd.read_hdf(path, gs, where=range_sel.query())
        f['feature'] = anno
        d.append(f)
    d = pd.concat(d)

    return d


def select_scores(path, dataset, range_sel, annos=None):
    if annos is None or type(annos) is bool:
        annos = hdf.ls(path, pt.join(dataset, 'scores'))
    d = []
    for anno in annos:
        gs = pt.join(dataset, 'scores', anno, range_sel.chromo)
        f = pd.read_hdf(path, gs, where=range_sel.query())
        f['feature'] = anno
        d.append(f)
    d = pd.concat(d)
    return d


class Selector(object):

    def __init__(self, features):
        self.features = features
        self.chromos = None
        self.start = None
        self.end = None
        self.samples = None
        self.spread = True
        self.logger = None
        self.dtype = 'float16'

    def log(self, msg):
        if self.logger is not None:
            self.logger(msg)

    def select_chromo(self, path, dataset, chromo):
        range_sel = RangeSelection(chromo, self.start, self.end)

        self.__tc = None
        def add_to_store(d, name):
            self.log('Store ...')
            store = self.__tc
            d['cat'] = name
            d = pd.pivot_table(d, index='pos', columns=['cat', 'feature'], values='value')
            if store is None:
                store = d
            else:
                store = pd.concat([store, d], axis=1)
            store = store.astype(self.dtype)
            self.__tc = store

        if self.features.cpg:
            self.log('cpg ...')
            df = select_cpg(path, dataset, range_sel, self.samples)
            add_to_store(df, 'cpg')

        if self.features.knn:
            self.log('knn ...')
            df = select_knn(path, dataset, range_sel, self.samples,
                            self.features.knn, False)
            add_to_store(df, 'knn')

        if self.features.knn_dist:
            self.log('knn_dist ...')
            df = select_knn(path, dataset, range_sel, self.samples,
                            self.features.knn, True)
            # log distance to avoid overflow for float16
            df['value'] = np.log2(df.value + 1)
            add_to_store(df, 'knn_dist')

        if self.features.annos:
            self.log('annos ...')
            df = select_annos(path, dataset, range_sel, self.features.annos)
            add_to_store(df, 'annos')

        if self.features.annos_dist:
            self.log('annos_dist ...')
            df = select_annos(path, dataset, range_sel, self.features.annos_dist, dist=True)
            # log distance to avoid overflow for float16
            df['value'] = np.log2(df.value + 1)
            add_to_store(df, 'annos_dist')

        if self.features.scores:
            self.log('scores ...')
            df = select_scores(path, dataset, range_sel, self.features.scores)
            # fill missing values by mean
            df.value.fillna(df.value.mean())
            add_to_store(df, 'scores')

        return self.__tc

    def select(self, path, dataset):
        if self.chromos is None:
            self.chromos = data.list_chromos(path, dataset)
        if self.samples is None:
            self.samples = hdf.ls(path, pt.join(dataset, 'cpg', '1'))

        self.__t = None
        def add_to_store(d, chromo):
            self.log('Store ...')
            i = pd.MultiIndex.from_product([chromo, d.index.values],
                                           names=['chromo', 'pos'])
            d.index = i
            store = self.__t
            if store is None:
                store = d
            else:
                store = pd.concat([store, d])
            del self.__tc
            store = store.astype(self.dtype)
            self.__t = store

        for chromo in self.chromos:
            self.log('Chromosome %s ...' % (str(chromo)))
            d = self.select_chromo(path, dataset, str(chromo))
            add_to_store(d, chromo)

        return self.__t
