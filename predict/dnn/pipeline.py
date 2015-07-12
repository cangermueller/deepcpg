import numpy as np
import pandas as pd
import os.path as pt
import pickle

import predict.dnn.mt_dnn as dmt
import predict.predict as pred
import predict.evaluation as peval


class Pipeline(object):

    def __init__(self, params, train_X, train_Y, val_X=None, val_Y=None,
                 test_X=None, test_Y=None, train_ws=None, val_ws=None,
                 base_path='.'):
        self.params = params
        self.data = dict(
            train=(train_X, train_Y, train_ws),
            val=(val_X, val_Y, val_ws),
            test=(test_X, test_Y, None)
        )
        self.base_path = base_path
        self.logger = lambda x: print(x)

    def log(self, x):
        if (self.logger):
            self.logger(x)

    def get_data(self, dset, weight=False):
        X, Y, w = self.data[dset]
        if weight:
            return (X, Y, w)
        else:
            return (X, Y)

    def fit(self):
        self.log('\nFit model ...')
        model = dmt.MtDnn(self.params, logger=print)
        train_X, train_Y, train_ws = self.get_data('train', True)
        val_X, val_Y, val_ws = self.get_data('val', True)
        model.fit(train_X, train_Y, val_X, val_Y, train_ws, val_ws)
        self.model = model

        self.log('Save model ...')
        t = pt.join(self.base_path, 'model.pkl')
        with open(t, 'wb') as f:
            pickle.dump(model, f)

    def perf(self, dset='train'):
        self.log('\nEvaluate %s performance ...' % (dset))
        X, Y = self.get_data(dset)
        Z = self.model.predict(X)
        Z = pd.DataFrame(Z, index=Y.index, columns=Y.columns)
        if dset == 'test':
            t = pt.join(self.base_path, 'z.h5')
            Z.to_hdf(t, dset)
        p = pred.scores_frame(Y, Z, funs=peval.eval_funs)
        t = pt.join(self.base_path, 'perf_%s.csv' % (dset))
        p.to_csv(t, sep='\t', index=False)
        self.log(p)
        return (Z, p)

    def run(self):
        self.fit()
        self.perf('train')
        if self.data['val'][0] is not None:
            self.perf('val')
        if self.data['test'][0] is not None:
            self.perf('test')
