#!/usr/bin/env python

"""Compute summary statistics of data files.

Computes summary statistics of data files such as the number of samples or the
mean and variance of output variables.

Examples
--------

.. code:: bash

    dcpg_data_stats.py
        ./data/*.h5
"""

from __future__ import print_function
from __future__ import division

from collections import OrderedDict
import os
import sys

import argparse
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import six

from deepcpg import data as dat
from deepcpg.data import hdf


def get_output_stats(output):
    stats = OrderedDict()
    output = np.ma.masked_values(output, dat.CPG_NAN)
    stats['nb_tot'] = len(output)
    stats['nb_obs'] = np.sum(output != dat.CPG_NAN)
    stats['frac_obs'] = stats['nb_obs'] / stats['nb_tot']
    stats['mean'] = float(np.mean(output))
    stats['var'] = float(np.var(output))
    return stats


def plot_stats(stats):
    stats = stats.sort_values('frac_obs', ascending=False)
    stats = pd.melt(stats, id_vars=['output'], var_name='metric')
    #  stats = stats.loc[stats.metric.isin(['frac_obs', 'frac_one'])]
    #  stats.metric = stats.metric.str.replace('frac_obs', 'cov')
    #  stats.metric = stats.metric.str.replace('frac_one', 'met')
    grid = sns.FacetGrid(data=stats, col='metric', sharex=False)
    grid.map(sns.barplot, 'value', 'output')
    for ax in grid.axes.ravel():
        ax.set(xlabel='', ylabel='')
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
            description='Computes data statistics')
        p.add_argument(
            'data_files',
            nargs='+',
            help='Data files')
        p.add_argument(
            '-o', '--out_tsv',
            help='Write statistics to tsv file')
        p.add_argument(
            '-f', '--out_fig',
            help='Create output figure')
        p.add_argument(
            '--output_names',
            help='List of regex to filter outputs',
            nargs='+')
        p.add_argument(
            '--nb_sample',
            help='Maximum number of samples',
            type=int)
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

        output_names = dat.get_output_names(opts.data_files[0],
                                            regex=opts.output_names)
        stats = OrderedDict()
        for name in output_names:
            output = hdf.read(opts.data_files, 'outputs/%s' % name,
                              nb_sample=opts.nb_sample)
            output = list(output.values())[0]
            stats[name] = get_output_stats(output)
        tmp = []
        for key, value in six.iteritems(stats):
            tmp.append(pd.DataFrame(value, index=[key]))
        stats = pd.concat(tmp)
        stats.index.name = 'output'
        stats.reset_index(inplace=True)

        print(stats.to_string())
        if opts.out_tsv:
            stats.to_csv(opts.out_tsv, sep='\t', index=False)

        if opts.out_fig:
            plot_stats(stats).savefig(opts.out_fig)

        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
