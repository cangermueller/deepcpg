import pandas as pd
import os
import os.path as pt
import pickle
import matplotlib
import matplotlib.pyplot as plt

import predict.dnn.mt_dnn as dmt
import predict.predict as pred
import predict.evaluation as peval


matplotlib.style.use('ggplot')


class Pipeline(object):

    def __init__(self, params, train_X, train_Y, val_X=None, val_Y=None,
                 test_X=None, test_Y=None, train_ws=None, val_ws=None,
                 base_path='.', logger=None):
        self.params = params
        self.data = dict(
            train=(train_X, train_Y, train_ws),
            val=(val_X, val_Y, val_ws),
            test=(test_X, test_Y, None)
        )
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)
        self.logger = logger

    def log(self, x):
        if (self.logger):
            self.logger(x)

    def get_data(self, dset, weight=False):
        X, Y, w = self.data[dset]
        if weight:
            return (X, Y, w)
        else:
            return (X, Y)

    def lc_plot(self, lc):
        t = ['cost_train']
        if self.data['val'][0] is not None:
            t.append('cost_val')
        d = lc.loc[:, t]
        fig, ax = plt.subplots(figsize=(10, 6))
        d.plot(ax=ax, figsize=(10, 6))
        ax.set_xlabel('epoch')
        ax.set_ylabel('cost')
        return (fig, ax)

    def fit(self):
        self.log('\nFit model ...')
        model = dmt.MtDnn(self.params, logger=self.logger)
        train_X, train_Y, train_ws = self.get_data('train', True)
        val_X, val_Y, val_ws = self.get_data('val', True)
        lc = []
        model.fit(train_X, train_Y, val_X, val_Y, train_ws, val_ws,
                  lc_logger=lambda x: lc.append(x.mean()))
        self.model = model

        self.log('\nSave model ...')
        model.logger = None
        t = pt.join(self.base_path, 'model.pkl')
        with open(t, 'wb') as f:
            pickle.dump(model, f)

        self.log('\nSave learning curve ...')
        lc = pd.DataFrame(pd.concat(lc, axis=1)).T
        lc['epoch'] = range(lc.shape[0])
        t = pt.join(self.base_path, 'lc.csv')
        lc.to_csv(t, sep='\t', index=0)
        t = pt.join(self.base_path, 'lc.pdf')
        fig, ax = self.lc_plot(lc)
        fig.savefig(t)

        return lc

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
