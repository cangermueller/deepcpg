#!/usr/bin/env python

"""Evaluate prediction performance.

Evaluates prediction performances globally and genomic annotations.

Examples
--------
Evaluate prediction performance globally and in genomic contexts annotated as
CGI, TSS, or gene body. Also compute precision recall and ROC curve of
individual outputs:

.. code:: bash

    dcpg_eval_perf.py
        ./eval/data.h5
        --out_dir ./eval
        --curves pr roc
        --annos_files ./bed/CGI.bed ./bed/TSS.bed ./bed/gene_body.bed
"""

import os
import sys

import argparse
import logging
import numpy as np
import pandas as pd
from sklearn import metrics as skm

from deepcpg import data as dat
from deepcpg import evaluation as ev
from deepcpg.data import hdf
from deepcpg.data.annotations import is_in, join_overlapping_frame
from deepcpg.utils import fold_dict, make_dir, slice_dict, to_list


ANNO_GLOBAL = 'global'


def annotate(chromos, pos, anno):
    """Annotate genomic locations.

    Tests if sites specified by `chromos` and `pos` are annotated by `anno`.

    Parameters
    ----------
    chromos: :class:`numpy.ndarray`
        :class:`numpy.ndarray` with chromosome of sites.
    pos: :class:`numpy.ndarray`
        :class:`numpy.ndarray` with position on chromosome of sites.
    anno: :class:`pandas.DataFrame`
        :class:`pandas.DataFrame` with columns `chromo`, `start`, `end` that
        specify annotated regions.

    Returns
    -------
    :class:`numpy.ndarray`
        Binary :class:`numpy.ndarray` of same length as `chromos` indicating if
        positions are annotated.
    """
    idx = []
    for chromo in np.unique(chromos):
        chromo_pos = pos[chromos == chromo]
        chromo_anno = anno.loc[anno.chromo == chromo]
        chromo_idx = is_in(chromo_pos,
                           chromo_anno['start'].values,
                           chromo_anno['end'].values)
        idx.append(chromo_idx)
    idx = np.hstack(idx)
    return idx


def read_anno_file(anno_file, chromos=None, nb_sample=None):
    """Read annotations from BED file.

    Reads annotations from BED file merges overlapping annotations.

    Parameters
    ----------
    anno_file: str
        File name.
    chromos: list
        List of chromosomes for filtering annotations.
    nb_sample: int
        Maximum number of annotated regions.

    Returns
    -------
    :class:`pandas.DataFrame`
        :class:`pandas.DataFrame` with columns `chromo`, `start`, `end`.
    """
    anno = pd.read_table(anno_file, header=None, usecols=[0, 1, 2],
                         dtype={0: 'str', 1: 'int32', 2: 'int32'},
                         nrows=nb_sample)
    anno.columns = ['chromo', 'start', 'end']
    anno.chromo = anno.chromo.str.upper().str.replace('chr', '', case=False)
    if chromos is not None:
        chromos = to_list(chromos)
        anno = anno.loc[anno.chromo.isin(chromos)]
    anno = join_overlapping_frame(anno)
    return anno


def get_curve_fun(name):
    """Return performance curve function by its name."""
    if name == 'roc':
        return skm.roc_curve
    elif name == 'pr':
        return skm.precision_recall_curve
    else:
        raise ValueError('Invalid performance curve "%s"!' % name)


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
            description='Evaluates predictions')
        p.add_argument(
            'data_file',
            help='HDF5 file from `dcpg_eval.py` with outputs and predictions')
        p.add_argument(
            '-o', '--out_dir',
            help='Output directory',
            default='.')
        p.add_argument(
            '--output_names',
            help='Regex to select outputs',
            nargs='+')
        p.add_argument(
            '--nb_output',
            help='Maximum number of outputs',
            type=int)
        p.add_argument(
            '--curves',
            help='Performance curves to be computed',
            nargs='+',
            choices=['roc', 'pr'])
        p.add_argument(
            '--nb_curve_point',
            help='Maximum number of points on performance curves',
            type=int,
            default=1000)
        p.add_argument(
            '--anno_files',
            help='BED files with annotation tracks',
            nargs='+')
        p.add_argument(
            '--anno_curves',
            help='Performance curves to be computed in annotations contexts',
            nargs='+',
            choices=['roc', 'pr'])
        p.add_argument(
            '--anno_min_sites',
            help='Minimum number of annotated sites required',
            default=100,
            type=int)
        p.add_argument(
            '--nb_sample',
            help='Maximum number of samples',
            type=int)
        p.add_argument(
            '--compress',
            help='Compress output files to reduce storage',
            action='store_true')
        p.add_argument(
            '--verbose',
            help='More detailed log messages',
            action='store_true')
        p.add_argument(
            '--log_file',
            help='Write log messages to file')
        return p

    def save_report(self, report, name, *args, **kwargs):
        filename = os.path.join(self.opts.out_dir, '%s.tsv' % name)
        compression = None
        if self.opts.compress:
            filename = '%s.gz' % filename
            compression = 'gzip'
        self.log.info('Writing %s ...' % filename)
        report.to_csv(filename, sep='\t', float_format='%.5f', index=False,
                      compression=compression, *args, **kwargs)

    def main(self, name, opts):
        logging.basicConfig(filename=opts.log_file,
                            format='%(levelname)s (%(asctime)s): %(message)s')
        log = logging.getLogger(name)
        if opts.verbose:
            log.setLevel(logging.DEBUG)
        else:
            log.setLevel(logging.INFO)
        log.debug(opts)

        self.opts = opts
        self.log = log

        # Get performance curve functions from names.
        curve_funs = dict()
        if opts.curves:
            for name in opts.curves:
                curve_funs[name] = get_curve_fun(name)
        anno_curve_funs = dict()
        if opts.anno_curves:
            for name in opts.anno_curves:
                anno_curve_funs[name] = get_curve_fun(name)

        log.info('Loading data ...')
        # Read and sort predictions and outputs.
        output_names = dat.get_output_names(opts.data_file,
                                            regex=opts.output_names,
                                            nb_key=opts.nb_output)
        names = {'chromo': None, 'pos': None,
                 'outputs': output_names,
                 'preds': output_names}
        data = hdf.read(opts.data_file, names, nb_sample=opts.nb_sample)
        data['chromo'] = [chromo.decode() for chromo in data['chromo']]
        data['chromo'] = np.array(data['chromo'])
        data = fold_dict(data, nb_level=1)
        idx = np.lexsort((data['pos'], data['chromo']))
        data = slice_dict(data, idx)
        for chromo in np.unique(data['chromo']):
            chromo_pos = data['pos'][data['chromo'] == chromo]
            tmp = np.sort(chromo_pos)
            assert np.all(chromo_pos == tmp)
        log.info('%d samples' % len(data['pos']))

        reports = []
        curves = []

        log.info('Evaluating globally ...')
        # Evaluate performances globally.
        report = ev.evaluate_outputs(data['outputs'], data['preds'])
        report['anno'] = ANNO_GLOBAL
        reports.append(report)
        pd.set_option('display.width', 1000)
        print(ev.unstack_report(report))

        if curve_funs:
            # Performance curves.
            for name, fun in curve_funs.items():
                log.info('%s curve' % name)
                curve = ev.evaluate_curve(data['outputs'], data['preds'],
                                          fun=fun, nb_point=opts.nb_curve_point)
                if curve is not None:
                    curve['curve'] = name
                    curve['anno'] = ANNO_GLOBAL
                    curves.append(curve)

        if opts.anno_files:
            log.info('Evaluating annotations ...')
            # Evaluate annotations.
            for anno_file in opts.anno_files:
                anno = read_anno_file(anno_file)
                anno_name = os.path.splitext(os.path.basename(anno_file))[0]
                idx = annotate(data['chromo'], data['pos'], anno)
                log.info('%s: %d' % (anno_name, idx.sum()))
                if idx.sum() < opts.anno_min_sites:
                    log.info('Skipping due to insufficient annotated sites!')
                    continue
                # Select data at annotated sites.
                anno_data = slice_dict(data, idx)
                report = ev.evaluate_outputs(anno_data['outputs'],
                                             anno_data['preds'])
                report['anno'] = anno_name
                reports.append(report)

                if curve_funs:
                    # Performance curves.
                    for name, fun in anno_curve_funs.items():
                        log.info('%s curve' % name)
                        curve = ev.evaluate_curve(
                            data['outputs'], data['preds'],
                            fun=fun, nb_point=opts.nb_curve_point)
                        if curve is not None:
                            curve['curve'] = name
                            curve['anno'] = anno_name
                            curves.append(curve)

        make_dir(opts.out_dir)
        if reports:
            report = pd.concat(reports)
            report = report[['anno', 'metric', 'output', 'value']]
            self.save_report(report, 'metrics')
        if curves:
            curves = pd.concat(curves)
            curves = curves[['anno', 'curve', 'output', 'x', 'y', 'thr']]
            self.save_report(curves, 'curves')

        log.info('Done!')

        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
