#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import pandas as pd
import re

__dir = pt.dirname(pt.realpath(__file__))
# sys.path.insert(0, pt.join(__dir, '../module'))

import hdf
import data


class FeatureSelection(object):

    def __init__(self, knn=5, knn_dist=False, annos=True):
        self.knn = knn  # k tor True
        self.knn_dist = knn_dist  # k or True
        self.annos = annos  # non-empty list or True


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


def select_annos(path, dataset, range_sel, annos=None):
    if annos is None or type(annos) is bool:
        annos = hdf.ls(path, pt.join(dataset, 'annos'))
    d = []
    for anno in annos:
        gs = pt.join(dataset, 'annos', anno, range_sel.chromo)
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

        df = select_cpg(path, dataset, range_sel, self.samples)
        pos = df.pos.unique()
        d = {'cpg': df}
        if self.features.knn:
            df = select_knn(path, dataset, range_sel, self.samples,
                            self.features.knn, False)
            df = df.loc[df.pos.isin(pos)]
            d['knn'] = df
        if self.features.knn_dist:
            df = select_knn(path, dataset, range_sel, self.samples,
                            self.features.knn, True)
            df = df.loc[df.pos.isin(pos)]
            d['knn_dist'] = df
        if self.features.annos:
            df = select_annos(path, dataset, range_sel, self.features.annos)
            df = df.loc[df.pos.isin(pos)]
            d['annos'] = df
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


class App(object):

    def run(self, args):
        name = pt.basename(args[0])
        parser = self.create_parser(name)
        opts = parser.parse_args(args[1:])
        return self.main(name, opts)

    def create_parser(self, name):
        p = argparse.ArgumentParser(
            prog=name,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Extract feature matrix from database')
        p.add_argument(
            'in_file',
            help='HDF path to dataset')
        p.add_argument(
            '-o', '--out_file',
            help='Output HDF path')
        p.add_argument(
            '-k', '--knn',
            help='Include knn CpG (int)',
            type=int)
        p.add_argument(
            '--knn_dist',
            help='Include distance to knn CpG (int)',
            type=int)
        p.add_argument(
            '--annos',
            help='Include annotations (True or list)',
            nargs='*')
        p.add_argument(
            '--chromos',
            help='Only use these chromosomes',
            nargs='+'),
        p.add_argument(
            '--samples',
            help='Only use these samples',
            nargs='+'),
        p.add_argument(
            '--start',
            help='Start position chromosome',
            type=int),
        p.add_argument(
            '--stop',
            help='Stop position chromosome',
            type=int),
        p.add_argument(
            '--verbose',
            help='More detailed log messages',
            action='store_true')
        p.add_argument(
            '--log_file',
            help='Write log messages to file')
        return p

    def main(self, name, opts):
        logging.basicConfig(filename=opts.log_file,
                            format='%(levelname)s (%(asctime)s): %(message)s')
        log = logging.getLogger(name)
        if opts.verbose:
            log.setLevel(logging.DEBUG)
        else:
            log.setLevel(logging.INFO)
        log.debug(opts)

        if opts.annos is not None and len(opts.annos) == 0:
            opts.annos = True

        fs = FeatureSelection()
        fs.knn = opts.knn
        fs.knn_dist = opts.knn_dist
        fs.annos = opts.annos

        sel = Selector(fs)
        sel.chromos = opts.chromos
        sel.samples = opts.samples
        sel.start = opts.start
        sel.end = opts.stop

        path, group = hdf.split_path(opts.in_file)
        log.info('Select ...')
        d = sel.select(path, group)
        print(d.columns.values)

        log.info('Write output ...')
        path, group = hdf.split_path(opts.out_file)
        d.to_hdf(path, group)

        log.info('Done!')
        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
