import sys
import os.path as pt
import numpy as np
import numpy.testing as npt
import pandas as pd
from sklearn.linear_model import LogisticRegression
import ipdb
import pytest

__dir = pt.dirname(pt.realpath(__file__))
sys.path.insert(0, pt.join(__dir, '../predict'))

import predict as pred



class TestMultiClassClassifier(object):

    def test_predict(self):
        rng = np.random.RandomState(0)
        X = rng.normal(0, 1, (100, 10))
        Y = rng.binomial(1, 0.5, (100, 5))
        Y = np.array(Y, dtype='float32')
        for i in range(Y.shape[1]):
            Y[rng.binomial(1, 0.5, Y.shape[0]) == 1, i] = np.nan

        mult = pred.MultitaskClassifier(LogisticRegression())
        mult.fit(X, Y)
        Z = mult.predict_proba(X)

        single = LogisticRegression()
        Xs = X
        Ys = Y[:, 2]
        h = ~np.isnan(Ys)
        Xs = X[h]
        Ys = Ys[h]
        single.fit(Xs, Ys)
        Zs = single.predict_proba(X)[:, 1]
        npt.assert_array_almost_equal(Z[:, 2], Zs)
        npt.assert_array_almost_equal(mult.models[2].coef_, single.coef_)


class TestSampleSpecificClassifier(object):

    def setup_class(self):
        self.rng = np.random.RandomState(0)

    def make_data(self, num_samples=3, num_feat=10, num_rows=100, num_shared=2):
        X = self.rng.normal(0, 1, (num_rows, (num_samples + num_shared) * num_feat))
        t = []
        for i in range(num_samples + 2):
            t.extend([i] * num_feat)
        t = pd.MultiIndex.from_arrays((t, list(range(num_feat)) * (num_samples + num_shared)))
        X = pd.DataFrame(X, columns=t)

        Y = self.rng.binomial(1, 0.5, (num_rows, num_samples))
        Y = np.array(Y, dtype='float32')
        for i in range(Y.shape[1]):
            t = self.rng.binomial(1, 0.5, Y.shape[0])
            Y[t == 1, i] = np.nan
        Y = pd.DataFrame(Y, columns=range(num_samples))

        return (X, Y)


    def test_X_(self):
        X, Y = self.make_data(5, 10, 100, 2)
        m = pred.SampleSpecificClassifier(None)
        m.shared = [5, 6]
        for i in range(5):
            Xi = m.X_(X, i)
            assert Xi.shape == (100, 30)
        a = X.loc[:, [3, 5, 6]]
        b = m.X_(X, 3)
        assert (a == b).all().all()


    def test_predict(self):
        X, Y = self.make_data()

        ss = pred.SampleSpecificClassifier(LogisticRegression())
        ss.fit(X, Y)
        Z = ss.predict_proba(X)

        sample = 1
        Xs = X[sample]
        Ys = Y[sample]
        h = ~Ys.isnull()
        Xs = Xs.loc[h]
        Ys = Ys.loc[h]
        s = LogisticRegression()
        s.fit(Xs.values, Ys.values)
        Zs = s.predict_proba(X[sample])[:, 1]
        npt.assert_array_almost_equal(Z[:, sample], Zs)
        npt.assert_array_almost_equal(ss.models[sample].coef_, s.coef_)
