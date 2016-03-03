#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import pandas as pd
import numpy as np


def model_name(train_dir):
    return pt.basename(pt.dirname(train_dir))


def read_file(dnames, fname, exists=False):
    ds = []
    for dname in dnames:
        h = pt.join(dname, fname)
        if not exists and not pt.isfile(h):
            continue
        d = pd.read_table(h)
        d['model'] = model_name(dname)
        ds.append(d)
    ds = pd.concat(ds)
    return ds


def read_lc(dnames, fname='lc.csv', exists=False):
    d = read_file(dnames, fname, exists)
    return d.groupby('model', as_index=False).last()


def read_perf(dnames, fname='perf_val.csv', exists=False, targets=None):
    d = read_file(dnames, fname, exists)
    if targets is not None:
        d = d.loc[d.target.isin(targets)]
    d = d.groupby('model', as_index=False).mean()
    d = d.loc[:, d.columns != 'loss']
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
            description='Inspect model training')
        p.add_argument(
            'train_dirs',
            help='Training directories',
            nargs='+')
        p.add_argument(
            '-r', '--regression',
            help='Evaluate regression performance',
            action='store_true')
        p.add_argument(
            '-s', '--sort',
            help='Sort ascending',
            default='val_loss')
        p.add_argument(
            '-S', '--Sort',
            help='Sort descending')
        p.add_argument(
            '--early',
            help='Filter early stopped',
            type=int,
            choices=[0, 1])
        p.add_argument(
            '--epoch',
            help='Minimum number of epochs',
            type=int)
        p.add_argument(
            '--loss',
            help='Minimum loss',
            type=float)
        p.add_argument(
            '--lc_file',
            help='Filename learning curve',
            default='lc.csv')
        p.add_argument(
            '--val_file',
            help='Filename validation performance')
        p.add_argument(
            '--targets',
            help='Filter targets',
            nargs='+')
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

        if opts.seed is not None:
            np.random.seed(opts.seed)

        if opts.val_file is None:
            if opts.regression:
                opts.val_file = 'val_reg.csv'
            else:
                opts.val_file = 'val_cla.csv'

        d = read_lc(opts.train_dirs, fname=opts.lc_file)
        models = [model_name(x) for x in opts.train_dirs]
        for m in filter(lambda x: not np.any(d.model == x), models):
            log.warn('%s incomplete!' % m)

        v = read_perf(opts.train_dirs, fname=opts.val_file,
                      targets=opts.targets)
        d = pd.merge(d, v, on='model', how='left')
        if opts.early is not None:
            t = d['auc'].isnull()
            if opts.early == 0:
                t = ~t
            d = d.loc[t]
        if opts.epoch:
            d = d.loc[d.epoch >= opts.epoch]
        if opts.loss:
            d = d.loc[d.val_loss >= opts.loss]
        if opts.Sort:
            d.sort_values(opts.Sort, inplace=True, ascending=False)
        else:
            d.sort_values(opts.sort, inplace=True, ascending=True)
        t = d.to_string(index=False)
        print(t)

        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
