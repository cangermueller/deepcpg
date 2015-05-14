#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import numpy as np
import pandas as pd
import re

__dir = pt.dirname(pt.realpath(__file__))
# sys.path.insert(0, pt.join(__dir, '../module'))
import feature_extractor as fext
import utils as ut
import hdf


def filter_features(d, annos=True, dist=False, exclude=[]):
    if annos is False:
        exclude.append('^anno_')
    if dist is False:
        exclude.append('^knn_dist')
    if len(exclude) > 0:
        h = d.feature.str.contains(exclude[0])
        for e in exclude[1:]:
            h |= d.feature.str.contains(e)
        d = d.loc[~h]
    return d


def feature_matrix(d):
    d = pd.pivot_table(d, index=['chromo', 'pos'],
                       columns=['sample', 'feature'], values='value')
    return d


def split_Xy(d, sample):
    h = ~d[(sample, 'cpg')].isnull()
    d = d.loc[h]
    d = d[[x for x in d.columns if x[0] == sample or not x[1].startswith('cpg')]]
    d = d.dropna()
    y = d[(sample, 'cpg')]
    x = d[[x for x in d.columns if x != (sample, 'cpg')]]
    return (x, y)


def split_XY(d):
    is_y = np.array([x[1] == 'cpg' for x in d.columns], dtype='bool')
    Y = d.loc[:, is_y]
    X = d.loc[:, ~is_y]
    h = ~X.isnull().any(axis=1)
    X = X.loc[h]
    Y = Y.loc[h]
    Y.columns = Y.columns.get_level_values(0)
    return (X, Y)


def read_txt(filename, nrows=None, max_pos=None):
    d = pd.read_table(filename, nrows=nrows)
    sample = d.columns[-1]
    d = d.iloc[:, [0, 1, 3]]
    d.columns = ['chromo', 'pos', 'value']
    d['sample'] = sample
    d['chromo'] = format_chromos(d['chromo'])
    if max_pos is not None:
        d = d.loc[d.pos <= max_pos]
    return d


def read_txts(filenames, *args, **kwargs):
    return pd.concat([read_txt(f, *args, **kwargs) for f in filenames])


def read_bed(filename):
    d = pd.read_table(filename, header=None)
    d = d.iloc[:, :3]
    d.columns = ['chromo', 'start', 'end']
    d['chromo'] = format_chromos(d['chromo'])
    return d


def read_annos(filenames):
    anno_tables = {}
    for anno_file in filenames:
        anno_table = read_bed(anno_file)
        anno_table = anno_table.sort(['chromo', 'start'])
        anno_name = pt.splitext(pt.basename(anno_file))[0]
        anno_tables[anno_name] = anno_table
    return anno_tables


def format_chromo(chromo):
    chromo = str(chromo)
    chromo = chromo.upper()
    chromo = re.sub('^CHR', '', chromo)
    return chromo


def format_chromos(chromos):
    return [format_chromo(x) for x in chromos]


def is_int(d):
    return d == round(d)


def size_to_int(size, n):
    if is_int(size):
        return size
    else:
        return round(size * n)


def train_test(d, test_size=0.5):
    """Splits rows of d into training and test set."""

    n = d.shape[0]
    test_size = size_to_int(test_size, n)
    idx = np.arange(n)
    test_idx = np.random.choice(idx, test_size, replace=False)
    test_idx = np.in1d(idx, test_idx)
    train = d.loc[~test_idx]
    test = d.loc[test_idx]
    return (train, test)


def train_test_val(d, test_size=0.5, val_size=0.1):
    """Splits rows of d into training, test, and validation set."""

    h, test = train_test(d, test_size)
    train, val_ = train_test(h, val_size)
    return (train, test, val_)


def get_pos(d):
    """Return common positions for of all samples."""

    return pd.pivot_table(d, index=['chromo', 'pos'], columns='sample',
                          values='value').reset_index()['pos'].values


def knn_features_sample(d, fe, pos, dist):
    """Computes KNN features of one sample and chromosome."""

    f = fe.extract(pos, d['pos'].values, d['value'].values)
    f = pd.DataFrame(f, columns=fe.labels)
    f['pos'] = pos
    f = pd.melt(f, id_vars='pos', var_name='feature', value_name='value')
    if dist is False:
        f = f.loc[~f.feature.str.contains('dist_')]
    return f


def knn_features_chromo(d, fe, dist):
    """Computes KNN feature of all samples and one chromosome."""

    pos = get_pos(d)
    return ut.group_apply(d, 'sample', knn_features_sample, fe=fe, pos=pos,
                          dist=dist)


def knn_features(d, k=5, dist=True):
    """Computes KNN features of all samples and chromosomes."""

    fe = fext.KnnCpgFeatureExtractor(k)
    return ut.group_apply(d, 'chromo', knn_features_chromo, fe=fe, dist=dist)


def anno_features_chromo(d, annos, fe):
    """Computes annotation features of one chromosome."""

    pos = get_pos(d)
    f_all = []
    for anno_name, anno_table in annos.items():
        start, end = fe.join_intervals(anno_table['start'].values,
                                       anno_table['end'].values)
        f = fe.extract(pos, start, end)
        f = pd.DataFrame(dict(pos=pos, feature=anno_name, value=f))
        f_all.append(f)
    f_all = pd.concat(f_all)
    return f_all


def anno_features(d, annos):
    """Computes annotation features of all chromosomes."""

    fe = fext.IntervalFeatureExtractor()
    return ut.group_apply(d, 'chromo', anno_features_chromo, annos=annos, fe=fe)


def features(d, k=5, dist=True, annos=None):
    """Computes KNN and annotatioin features. d is CpG table."""

    f_all = []
    f = d.copy()
    f['feature'] = 'cpg'
    f_all.append(f)
    if k is not None:
        f = knn_features(d, k, dist)
        f['feature'] = ['knn_' + x for x in f['feature']]
        f_all.append(f)

    if annos is not None:
        f = anno_features(d, annos)
        f['sample'] = 'global'
        f['feature'] = ['anno_' + x for x in f['feature']]
        f_all.append(f)
    f_all = pd.concat(f_all)
    return f_all


class Data(object):

    def run(self, args):
        name = pt.basename(args[0])
        parser = self.create_parser(name)
        opts = parser.parse_args(args[1:])
        return self.main(name, opts)

    def create_parser(self, name):
        p = argparse.ArgumentParser(
            prog=name,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Computes training, test, and validation set.')
        p.add_argument(
            'in_files',
            help='Input CpG txt files',
            nargs='+')
        p.add_argument('-o', '--out_file',
                       help='Output HDF path. Creates train, test, val groups.',
                       default='data.h5')
        p.add_argument('--anno_files',
                       help='Annotation files in BED format.',
                       nargs='+')
        p.add_argument('-k', '--knn',
                       help='Number of K nearest neighbors',
                       type=int,
                       default=5)
        p.add_argument('--test_size',
                       help='Size of test set.',
                       type=float,
                       default=0.5)
        p.add_argument('--val_size',
                       help='Size of validation set.',
                       type=float,
                       default=0.1)
        p.add_argument('--nrows',
                       help='Read only that many rows',
                       type=int)
        p.add_argument('--max_pos',
                       help='Maximum position on chromosome',
                       type=int)
        p.add_argument(
            '--no_dist',
            help='Exclude KNN distance',
            action='store_true')
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

        log.info('Read CpG files ...')
        self.cpgs = read_txts(opts.in_files, nrows=opts.nrows,
                              max_pos=opts.max_pos)

        if opts.anno_files is None:
            self.annos = None
        else:
            log.info('Read annotation files ...')
            self.annos = read_annos(opts.anno_files)

        log.info('Split into train, test, val ...')
        self.cpgs = train_test_val(self.cpgs)
        names = ['train', 'test', 'val']
        self.cpgs = dict(zip(names, self.cpgs))

        hdf_file, hdf_path = hdf.split_path(opts.out_file)
        log.info('Process data sets ...')
        for k in names:
            log.info('\t%s' % (k))
            log.info('\t\t-> compute features')
            f = features(self.cpgs[k], k=opts.knn, dist=not opts.no_dist,
                         annos=self.annos)
            log.info('\t\t-> write to file')
            f.to_hdf(hdf_file, pt.join(hdf_path, k))

        log.info('Done!')
        return 0


if __name__ == '__main__':
    app = Data()
    app.run(sys.argv)
