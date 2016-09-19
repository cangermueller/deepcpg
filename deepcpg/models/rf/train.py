#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import h5py as h5
import numpy as np
import pandas as pd
import pickle
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import RandomizedSearchCV
from sklearn.cross_validation import PredefinedSplit
import scipy.stats as sps
from predict.evaluation import evaluate


def read_data(path, max_samples=None):
    f = h5.File(path)
    d = dict()
    for k in ['X', 'y', 'columns']:
        g = f[k]
        if max_samples is None:
            d[k] = g.value
        else:
            d[k] = g[:max_samples]
    f.close()
    return d['X'], d['y'], d['columns']

def eval_io(m, X, y, name, out_dir):
    z = m.predict_proba(X)[:, 1]
    p = evaluate(y, z)
    p = p.to_csv(None, sep='\t', index=False, float_format='%.4f')
    print('%s set performance:' % (name))
    print(p)
    with open(pt.join(out_dir, 'perf_%s.csv' % (name)), 'w') as f:
        f.write(p)

def cvgrid_to_frame(grid):
    mean = []
    params = []
    for l in grid:
        mean.append(l.mean_validation_score)
        params.append(l.parameters)
    d = {k: [] for k in params[0].keys()}
    for param in params:
        for k, v in param.items():
            d[k].append(v)
    d['mean'] = mean
    d = pd.DataFrame(d, columns=['mean'] + list(params[0].keys()))
    d.sort_values('mean', ascending=False, inplace=True)
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
            description='Train random forest classifier')
        p.add_argument(
            'train_data',
            help='HDF file with training data')
        p.add_argument(
            '--val_data',
            help='HDF file with training data')
        p.add_argument(
            '-o', '--out_dir',
            help='Output directory')
        p.add_argument(
            '--num_cv',
            help='Number of cross-validation iterations',
            type=int,
            default=1)
        p.add_argument(
            '--num_folds',
            help='Number of cross-validation folds if not validation data given',
            type=int,
            default=2)
        p.add_argument(
            '--num_cores',
            help='Number of CPU cores',
            type=int,
            default=1)
        p.add_argument(
            '--max_samples',
            help='Limit # samples',
            type=int)
        p.add_argument(
            '--seed',
            help='Seed of rng',
            type=int,
            default=0)
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

        log.info('Read data')
        X, y, *_ = read_data(opts.train_data, opts.max_samples)
        num_train = X.shape[0]
        print('Train data: %d x %d' % X.shape)

        if opts.val_data:
            log.info('Read val')
            X_val, y_val, *_ = read_data(opts.val_data, opts.max_samples)
            num_val = X_val.shape[0]
            print('Val data: %d x %d' % X_val.shape)
            X = np.vstack((X, X_val))
            y = np.hstack((y, y_val))
            del X_val
            del y_val
        else:
            num_val = 0

        if num_val == 0:
            cv = opts.num_folds
        else:
            split = np.zeros(X.shape[0], dtype='int8')
            split[:num_train] = -1
            cv = PredefinedSplit(split)

        log.info('Fit model')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = RandomForestClassifier()
            params = {
                'max_depth': sps.randint(5, 15),
                'n_estimators': sps.randint(5, 15)
            }
            m = RandomizedSearchCV(m, params, cv=cv,
                            n_iter=opts.num_cv,
                            scoring='roc_auc',
                            n_jobs=opts.num_cores)
            m.fit(X, y)
        with open(pt.join(opts.out_dir, 'model.pkl'), 'wb') as f:
            pickle.dump(m.best_estimator_, f)

        print('Cross-validation scores:')
        scores = cvgrid_to_frame(m.grid_scores_)
        s = scores.to_csv(None, sep='\t', index=False, float_format='%.3f')
        print(s)
        with open(pt.join(opts.out_dir, 'cv_scores.csv'), 'w') as f:
            f.write(s)

        print('Best model:')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            print(m.best_estimator_)

        log.info('Done!')

        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
