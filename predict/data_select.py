import argparse
import sys
import logging
import os.path as pt
import pandas as pd
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

    def select_chromo(self, path, dataset, chromo):
        range_sel = RangeSelection(chromo, self.start, self.end)

        d = dict()
        pos = None  # prefilter to reduce memory usage
        if self.features.cpg:
            df = select_cpg(path, dataset, range_sel, self.samples)
            pos = df.pos.unique()
            d = {'cpg': df}
        if self.features.knn:
            df = select_knn(path, dataset, range_sel, self.samples,
                            self.features.knn, False)
            if pos is not None:
                df = df.loc[df.pos.isin(pos)]
            d['knn'] = df
        if self.features.knn_dist:
            df = select_knn(path, dataset, range_sel, self.samples,
                            self.features.knn, True)
            if pos is not None:
                df = df.loc[df.pos.isin(pos)]
            d['knn_dist'] = df
        if self.features.annos:
            df = select_annos(path, dataset, range_sel, self.features.annos)
            if pos is not None:
                df = df.loc[df.pos.isin(pos)]
            d['annos'] = df
        if self.features.annos_dist:
            df = select_annos(path, dataset, range_sel, self.features.annos_dist, dist=True)
            if pos is not None:
                df = df.loc[df.pos.isin(pos)]
            d['annos_dist'] = df
        if self.features.scores:
            df = select_scores(path, dataset, range_sel, self.features.scores)
            if pos is not None:
                df = df.loc[df.pos.isin(pos)]
            d['scores'] = df
        for k, v, in d.items():
            d[k]['cat'] = k
        d = pd.concat(d)
        return d

    def select(self, path, dataset):
        if self.chromos is None:
            self.chromos = data.list_chromos(path, dataset)
        if self.samples is None:
            self.samples = hdf.ls(path, pt.join(dataset, 'cpg', '1'))
        d = []
        for chromo in self.chromos:
            chromo = str(chromo)
            dc = self.select_chromo(path, dataset, chromo)
            dc['chromo'] = chromo
            d.append(dc)
        d = pd.concat(d)
        if self.spread:
            d = pd.pivot_table(d, index=['chromo', 'pos'],
                               columns=['cat', 'feature'], values='value')
        return d
