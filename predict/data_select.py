import os.path as pt
import pandas as pd
import numpy as np
import re

from predict import data, hdf


class FeatureSelection(object):

    def __init__(self):
        self.cpg = False  # True, False, or list ['train', 'test', 'val']
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


def select_cpg(path, dataset, range_sel, samples=None, subsets=None):
    if subsets is None:
        subsets = ['']
    if samples is None:
        g = pt.join(dataset, subsets[0], 'cpg', range_sel.chromo)
        samples = hdf.ls(path, g)
    d = None
    for sample in samples:
        f = []
        for group in subsets:
            g = pt.join(dataset, group, 'cpg', range_sel.chromo, sample)
            f.append(pd.read_hdf(path, g, where=range_sel.query()))
        f = pd.concat(f)
        f.columns = [sample]
        if d is None:
            d = f
        else:
            d = pd.concat((d, f), axis=1)
    return d



def select_knn(path, dataset, range_sel, samples=None, k=None, dist=False, log=None):
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
    d = None
    for sample in samples:
        if log is not None:
            log('  %s ...' % (sample))
        gs = pt.join(g, sample)
        f = pd.read_hdf(path, gs, where=range_sel.query())
        if dist:
            t = 'dist_'
        else:
            t = 'cpg_'
        f['feature'] = [sample + '_' + x.replace(t, '') for x in f.feature]
        f = pd.pivot_table(f, index='pos', columns='feature', values='value')
        if d is None:
            d = f
        else:
            e = pd.concat((d, f), axis=1)
            assert e.shape[0] == d.shape[0]
            d = e
    return d


def select_annos(path, dataset, range_sel, annos=None, dist=False, log=None):
    group = 'annos'
    if dist:
        group += '_dist'
    if annos is None or type(annos) is bool:
        annos = hdf.ls(path, pt.join(dataset, group))
    d = None
    for anno in annos:
        if log is not None:
            log('  %s ...' % (anno))
        gs = pt.join(dataset, group, anno, range_sel.chromo)
        f = pd.read_hdf(path, gs, where=range_sel.query())
        f.columns = [anno]
        if d is None:
            d = f
        else:
            e = pd.concat((d, f), axis=1)
            assert e.shape[0] == d.shape[0]
            d = e
    return d


def select_scores(path, dataset, range_sel, annos=None, log=None):
    if annos is None or type(annos) is bool:
        annos = hdf.ls(path, pt.join(dataset, 'scores'))
    d = None
    for anno in annos:
        if log is not None:
            log('  %s ...' % (anno))
        gs = pt.join(dataset, 'scores', anno, range_sel.chromo)
        f = pd.read_hdf(path, gs, where=range_sel.query())
        f.columns = [anno]
        if d is None:
            d = f
        else:
            e = pd.concat((d, f), axis=1)
            assert e.shape[0] == d.shape[0]
            d = e
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
        def add_to_store(d, name, pivot=False):
            if pivot:
                self.log('  pivot ...')
                d = pd.pivot_table(d, index='pos', columns='feature', values='value')
            d.columns = pd.MultiIndex.from_product((name, d.columns))
            self.log('  concat ...')
            store = self.__tc
            if store is None:
                store = d
            else:
                t = pd.concat([store, d], axis=1)
                assert t.shape[0] == store.shape[0]
                store = t
            store = store.astype(self.dtype)
            self.__tc = store

        if self.features.cpg:
            self.log('cpg ...')
            if isinstance(self.features.cpg, list):
                t = self.features.cpg
            else:
                t = None
            df = select_cpg(path, dataset, range_sel, self.samples, t)
            add_to_store(df, 'cpg')

        if self.features.knn:
            self.log('knn ...')
            df = select_knn(path, dataset, range_sel, self.samples,
                            self.features.knn, False, log=self.log)
            add_to_store(df, 'knn')

        if self.features.knn_dist:
            self.log('knn_dist ...')
            df = select_knn(path, dataset, range_sel, self.samples,
                            self.features.knn, True, log=self.log)
            # log distance to avoid overflow for float16
            df = np.log2(df + 1)
            add_to_store(df, 'knn_dist')

        if self.features.annos:
            self.log('annos ...')
            df = select_annos(path, dataset, range_sel, self.features.annos, log=self.log)
            assert np.all(df.notnull())
            add_to_store(df, 'annos')

        if self.features.annos_dist:
            self.log('annos_dist ...')
            df = select_annos(path, dataset, range_sel, self.features.annos_dist, dist=True, log=self.log)
            # log distance to avoid overflow for float16
            df = np.log2(df + 1)
            assert np.all(df.notnull())
            add_to_store(df, 'annos_dist')

        if self.features.scores:
            self.log('scores ...')
            df = select_scores(path, dataset, range_sel, self.features.scores, log=self.log)
            # fill missing values by mean
            df.fillna(df.mean(axis=0), inplace=True)
            assert np.all(df.notnull())
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
                t = pd.concat([store, d])
                assert t.shape[0] == store.shape[0]
                store = t
            del self.__tc
            store = store.astype(self.dtype)
            self.__t = store

        for chromo in self.chromos:
            self.log('Chromosome %s ...' % (str(chromo)))
            d = self.select_chromo(path, dataset, str(chromo))
            add_to_store(d, chromo)

        return self.__t


def select_cpg_matrix(path, group='/', chromo='1', subsets=None, reindex=False):
    fs = FeatureSelection()
    fs.cpg = subsets if subsets else True
    sel = Selector(fs)
    Y = sel.select(path, group)
    Y.index = Y.index.droplevel(0)
    if reindex:
        p = data.read_pos(path, group, chromo)
        Y = Y.reindex(p)
    return Y
