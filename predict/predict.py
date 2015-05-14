#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import numpy as np
import pandas as pd
import sklearn.metrics as met
import sklearn.base as skb
import ipdb

__dir = pt.dirname(pt.realpath(__file__))
# sys.path.insert(0, pt.join(__dir, '../module'))


def complete_cases(x, y=None):
    x = [x]
    if y is not None:
        x.append(y)
    x = [np.asarray(x_) for x_ in x]
    h = None
    for x_ in x:
        if len(x_.shape) == 1:
            hx = ~np.isnan(x_)
        else:
            hx = ~np.any(np.isnan(x_), axis=1)
        if h is None:
            h = hx
        else:
            h &= hx
    xc = [x_[h] for x_ in x]
    return xc


def score(Y, Yp, fun=met.roc_auc_score):
    y = np.asarray(Y).ravel()
    yp = np.asarray(Yp).ravel()
    y, yp = complete_cases(y, yp)
    return fun(y, yp)


def scores(Y, Z, fun=met.roc_auc_score):
    Y = np.asarray(Y)
    Z = np.asarray(Z)
    assert Y.shape == Z.shape
    num_tasks = Y.shape[1]
    scores = []
    for task in range(num_tasks):
        y = Y[:, task]
        z = Z[:, task]
        y, z = complete_cases(y, z)
        scores.append(fun(y, z))
    return scores


def format_write(Y, Z):
    assert Y.shape == Z.shape
    Z = pd.DataFrame(Z)
    Z.columns = Y.columns
    Z.index = Y.index
    Z = Z.reset_index()
    Z = pd.melt(Z, var_name='sample', value_name='value', id_vars=['chromo', 'pos'])
    Z['feature'] = 'z'
    return Z


def write_prediction(Y, Z, filename, group='z'):
    Z = format_write(Y, Z)
    Z.to_hdf(filename, group, format='t', data_columns=True)


class MultitaskClassifier(object):

    def __init__(self, m):
        self.model = m

    def Xy_(self, X, Y, task):
        y = Y[:, task]
        h = ~np.isnan(y)
        X = X[h]
        y = y[h]
        return (X, y)

    def fit(self, X, Y):
        X = np.asarray(X)
        Y = np.asarray(Y)
        self.num_tasks = Y.shape[1]
        self.models = []
        for task in range(self.num_tasks):
            Xt, yt = self.Xy_(X, Y, task)
            m = skb.clone(self.model)
            m.fit(Xt, yt)
            self.models.append(m)

    def predict(self, X):
        X = np.asarray(X)
        Y = []
        for task in range(self.num_tasks):
            m = self.models[task]
            Y.append(m.predict(X))
        Y = np.vstack(Y).T
        return Y

    def predict_proba(self, X):
        X = np.asarray(X)
        Y = []
        for task in range(self.num_tasks):
            m = self.models[task]
            Y.append(m.predict_proba(X)[:, 1])
        Y = np.vstack(Y).T
        return Y


class SampleSpecificClassifier(object):

    def __init__(self, m):
        self.model = m

    def X_(self, X, sample):
        X = X.loc[:, [sample] + self.shared]
        return X

    def Xy_(self, X, y, sample):
        X = self.X_(X, sample)
        y = y[sample].values
        return (X, y)

    def drop_na_(self, X, y):
        h = ~np.isnan(y)
        X = X.loc[h]
        y = y[h]
        return (X, y)

    def fit(self, X, Y):
        self.samples = Y.columns
        self.shared = [x not in self.samples for x in X.columns.get_level_values(0).unique()]
        self.models = []
        for sample in self.samples:
            Xs, ys = self.Xy_(X, Y, sample)
            Xs, ys = self.drop_na_(Xs, ys)
            ms = skb.clone(self.model)
            ms.fit(Xs, ys)
            self.models.append(ms)

    def predict(self, X):
        Y = []
        for i, sample in enumerate(self.samples):
            Xs = self.X_(X, sample)
            ms = self.models[i]
            Y.append(ms.predict(Xs))
        Y = np.vstack(Y).T
        return Y

    def predict_proba(self, X):
        Y = []
        for i, sample in enumerate(self.samples):
            Xs = self.X_(X, sample)
            ms = self.models[i]
            Y.append(ms.predict_proba(Xs)[:, 1])
        Y = np.vstack(Y).T
        return Y


class Predict(object):

    def run(self, args):
        name = pt.basename(args[0])
        parser = self.create_parser(name)
        opts = parser.parse_args(args[1:])
        return self.main(name, opts)

    def create_parser(self, name):
        p = argparse.ArgumentParser(
            prog=name,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Description')
        p.add_argument(
            'in_file',
            help='HDF path where chromosome seqs can be found')
        p.add_argument('-o', '--out_file',
                       help='Output file')
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

        return 0


if __name__ == '__main__':
    app = Predict()
    app.run(sys.argv)
