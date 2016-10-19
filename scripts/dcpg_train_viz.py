#!/usr/bin/env python

import os
import sys

import argparse
import logging
import matplotlib as mpl
mpl.use('agg')
import pandas as pd
import seaborn as sns


def plot_lc(lc, metrics=None):
    lc = pd.melt(lc, id_vars=['split', 'epoch'], var_name='output')
    if metrics:
        if not isinstance(metrics, list):
            metrics = [metrics]
        tmp = '(%s)' % ('|'.join(metrics))
        lc = lc.loc[lc.output.str.contains(tmp)]
    metrics = lc.output[~lc.output.str.contains('_')].unique()
    lc['metric'] = ''
    for metric in metrics:
        lc.loc[lc.output.str.contains(metric), 'metric'] = metric
        lc.loc[lc.output == metric, 'output'] = 'mean'
        lc.output = lc.output.str.replace('_%s' % metric, '')
        lc.output = lc.output.str.replace('cpg_', '')

    grid = sns.FacetGrid(lc, col='split', row='metric', hue='output',
                         sharey=False, size=3, aspect=1.2, legend_out=True)
    grid.map(mpl.pyplot.plot, 'epoch', 'value', linewidth=2)
    grid.set(ylabel='')
    grid.add_legend()
    return grid


class App(object):

    def run(self, args):
        name = os.path.basename(args[0])
        parser = self.create_parser(name)
        opts = parser.parse_args(args[1:])
        return self.main(name, opts)

    def create_parser(self, name):
        p = argparse.ArgumentParser(
            prog=name,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Visualizes training learning curves')
        p.add_argument(
            'train_lc',
            help='Training learning curve')
        p.add_argument(
            'val_lc',
            help='Validation learning curve')
        p.add_argument(
            '--metrics',
            help='Performance metrics to be visualized',
            nargs='+')
        p.add_argument(
            '-o', '--out_fig',
            help='Output figure',
            default='lc.pdf')
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

        lc = []
        for split, filename in zip(['train', 'val'],
                                   [opts.train_lc, opts.val_lc]):
            _lc = pd.read_table(filename)
            _lc['split'] = split
            _lc['epoch'] = range(1, len(_lc) + 1)
            lc.append(_lc)
        lc = pd.concat(lc)

        plot = plot_lc(lc, metrics=opts.metrics)
        plot.savefig(opts.out_fig)

        log.info('Done!')

        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
