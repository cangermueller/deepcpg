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

    def X_(self, X, task):
        return X[:, self.task_features[task]]

    def Xy_(self, X, Y, task):
        y = Y[:, task]
        h = ~np.isnan(y)
        y = y[h]
        X = self.X_(X, task)
        X = X[h]
        return (X, y)

    def fit(self, X, Y, task_features=None):
        X = np.asarray(X)
        Y = np.asarray(Y)
        self.ntasks = Y.shape[1]
        if task_features is None:
            task_features = [range(X.shape[1])] * self.ntasks
        self.task_features = task_features
        self.models = []
        for task in range(self.ntasks):
            Xt, yt = self.Xy_(X, Y, task)
            m = skb.clone(self.model)
            m.fit(Xt, yt)
            self.models.append(m)

    def predict(self, X):
        X = np.asarray(X)
        Y = []
        for task in range(self.ntasks):
            m = self.models[task]
            Xt = self.X_(X, task)
            Y.append(m.predict(Xt))
        Y = np.vstack(Y).T
        return Y

    def predict_proba(self, X):
        X = np.asarray(X)
        Y = []
        for task in range(self.ntasks):
            m = self.models[task]
            Xt = self.X_(X, task)
            Y.append(m.predict_proba(Xt)[:, 1])
        Y = np.vstack(Y).T
        return Y

    def get_params(self):
        return self.model.get_params()

    def set_params(self, **kwargs):
        return self.model.set_params(**kwargs)

    def coef_(self, order=False, abs_=True):
        fi = np.vstack([x.coef_ for x in self.models])
        if abs_:
            fi = np.abs(fi)
        if order:
            o = np.mean(fi, axis=0).argsort()
            return (fi, o)
        else:
            return fi

    def feature_importances_(self, order=False):
        fi = np.vstack([x.feature_importances_ for x in self.models])
        if order:
            o = np.mean(fi, axis=0).argsort()
            return (fi, o)
        else:
            return fi

    def feature_importances(self, *args, **kwargs):
        attr = None
        for a in ['feature_importances_', 'coef_']:
            if hasattr(self.models[0], a):
                attr = a
                break
        f = getattr(self, attr)
        return f(*args, **kwargs)


def flatten_index(index, sep='_'):
    if index.nlevels > 1:
        rv = [sep.join(x) for x in index.values]
    else:
        rv = index.values
    return rv


def sample_features(samples, features):
    ffeatures = flatten_index(features)
    shared = []
    specific = []
    for i, f in enumerate(ffeatures):
        is_shared = True
        for s in samples:
            if f.find(s) >= 0:
                is_shared = False
                break
        if is_shared:
            shared.append(i)
        else:
            specific.append(i)
    rv = []
    for s in samples:
        sf = []
        for i in specific:
            if ffeatures[i].find(s) >= 0:
                sf.append(i)
        sf.extend(shared)
        rv.append(sf)
    return rv

