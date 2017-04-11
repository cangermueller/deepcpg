#!/usr/bin/env python

"""Create DeepCpG input data from incomplete methylation profiles.

Takes as input incomplete CpG methylation profiles of multiple cells, extracts
neighboring CpG sites and/or DNA sequences windows, and writes data chunk files
to output directory. Output data can than be used for model training using
``dcpg_train.py`` model evaluation using ``dcpg_eval.py``.

Examples
--------
Create data files for training a CpG and DNA model, using 50 neighboring
methylation states and DNA sequence windows of 1001 bp from the mm10 genome
build:

.. code:: bash

    dcpg_data.py
        --cpg_profiles ./cpg/*.tsv
        --cpg_wlen 50
        --dna_files ./mm10
        --dna_wlen 1001
        --out_dir ./data

Create data files from gzip-compressed bedGraph files for predicting the mean
methylation rate and cell-to-cell variance from the DNA sequence:

.. code:: bash

    dcpg_data.py
        --cpg_profiles ./cpg/*.bedGraph.gz
        --dna_files ./mm10
        --dna_wlen 1001
        --win_stats mean var
        --win_stats_wlen 1001 2001 3001 4001 5001
        --out_dir ./data


See Also
--------
* ``dcpg_data_stats.py``: For computing statistics of data files.
* ``dcpg_data_show.py``: For showing the content of data files.
* ``dcpg_train.py``: For training a model.
"""

from __future__ import print_function
from __future__ import division

from collections import OrderedDict
import os
import sys
import warnings

import argparse
import logging
import h5py as h5
import numpy as np
import pandas as pd

import six
from six.moves import range

from deepcpg import data as dat
from deepcpg.data import annotations as an
from deepcpg.data import stats
from deepcpg.data import dna
from deepcpg.data import fasta
from deepcpg.data import feature_extractor as fext
from deepcpg.utils import make_dir


def prepro_pos_table(pos_tables):
    """Extracts unique positions and sorts them."""
    if not isinstance(pos_tables, list):
        pos_tables = [pos_tables]

    pos_table = None
    for next_pos_table in pos_tables:
        if pos_table is None:
            pos_table = next_pos_table
        else:
            pos_table = pd.concat([pos_table, next_pos_table])
        pos_table = pos_table.groupby('chromo').apply(
            lambda df: pd.DataFrame({'pos': np.unique(df['pos'])}))
        pos_table.reset_index(inplace=True)
        pos_table = pos_table[['chromo', 'pos']]
        pos_table.sort_values(['chromo', 'pos'], inplace=True)
    return pos_table


def split_ext(filename):
    """Remove file extension from `filename`."""
    return os.path.basename(filename).split(os.extsep)[0]


def read_cpg_profiles(filenames, log=None, *args, **kwargs):
    """Read methylation profiles.

    Input files can be gzip compressed.

    Returns
    -------
    dict
        `dict (key, value)`, where `key` is the output name and `value` the CpG
        table.
    """

    cpg_profiles = OrderedDict()
    for filename in filenames:
        if log:
            log(filename)
        cpg_file = dat.GzipFile(filename, 'r')
        output_name = split_ext(filename)
        cpg_profile = dat.read_cpg_profile(cpg_file, sort=True, *args, **kwargs)
        cpg_profiles[output_name] = cpg_profile
        cpg_file.close()
    return cpg_profiles


def extract_seq_windows(seq, pos, wlen, seq_index=1, assert_cpg=False):
    """Extracts DNA sequence windows at positions.

    Parameters
    ----------
    seq: str
        DNA sequence.
    pos: list
        Positions at which windows are extracted.
    wlen: int
        Window length.
    seq_index: int
        Offset at which positions start.
    assert_cpg: bool
        If `True`, check if positions in `pos` point to CpG sites.

    Returns
    -------
    np.array
        Array with integer-encoded sequence windows.
    """

    delta = wlen // 2
    nb_win = len(pos)
    seq = seq.upper()
    seq_wins = np.zeros((nb_win, wlen), dtype='int8')

    for i in range(nb_win):
        p = pos[i] - seq_index
        if p < 0 or p >= len(seq):
            raise ValueError('Position %d not on chromosome!' % (p + seq_index))
        if seq[p:p + 2] != 'CG':
            warnings.warn('No CpG site at position %d!' % (p + seq_index))
        win = seq[max(0, p - delta): min(len(seq), p + delta + 1)]
        if len(win) < wlen:
            win = max(0, delta - p) * 'N' + win
            win += max(0, p + delta + 1 - len(seq)) * 'N'
            assert len(win) == wlen
        seq_wins[i] = dna.char_to_int(win)
    # Randomly choose missing nucleotides
    idx = seq_wins == dna.CHAR_TO_INT['N']
    seq_wins[idx] = np.random.randint(0, 4, idx.sum())
    assert seq_wins.max() < 4
    if assert_cpg:
        assert np.all(seq_wins[:, delta] == 3)
        assert np.all(seq_wins[:, delta + 1] == 2)
    return seq_wins


def map_values(values, pos, target_pos, dtype=None, nan=dat.CPG_NAN):
    """Maps `values` array at positions `pos` to `target_pos`.

    Inserts `nan` for uncovered positions.
    """
    assert len(values) == len(pos)
    assert np.all(pos == np.sort(pos))
    assert np.all(target_pos == np.sort(target_pos))

    values = values.ravel()
    pos = pos.ravel()
    target_pos = target_pos.ravel()
    idx = np.in1d(pos, target_pos)
    pos = pos[idx]
    values = values[idx]
    if not dtype:
        dtype = values.dtype
    target_values = np.empty(len(target_pos), dtype=dtype)
    target_values.fill(nan)
    idx = np.in1d(target_pos, pos).nonzero()[0]
    assert len(idx) == len(values)
    assert np.all(target_pos[idx] == pos)
    target_values[idx] = values
    return target_values


def map_cpg_tables(cpg_tables, chromo, chromo_pos):
    """Maps values from cpg_tables to `chromo_pos`.

    Positions in `cpg_tables` for `chromo`  must be a subset of `chromo_pos`.
    Inserts `dat.CPG_NAN` for uncovered positions.
    """
    chromo_pos.sort()
    mapped_tables = OrderedDict()
    for name, cpg_table in six.iteritems(cpg_tables):
        cpg_table = cpg_table.loc[cpg_table.chromo == chromo]
        cpg_table = cpg_table.sort_values('pos')
        mapped_table = map_values(cpg_table.value.values,
                                  cpg_table.pos.values,
                                  chromo_pos)
        assert len(mapped_table) == len(chromo_pos)
        mapped_tables[name] = mapped_table
    return mapped_tables


def format_out_of(out, of):
    return '%d / %d (%.1f%%)' % (out, of, out / of * 100)


def get_stats_meta(names):
    funs = OrderedDict()
    for name in names:
        fun = stats.get(name)
        if name in ['mode', 'cat_var', 'cat2_var', 'diff']:
            dtype = np.int8
        else:
            dtype = np.float32
        funs[name] = (fun, dtype)
    return funs


def select_dict(data, idx):
    data = data.copy()
    for key, value in six.iteritems(data):
        if isinstance(value, dict):
            data[key] = select_dict(value, idx)
        else:
            data[key] = value[idx]
    return data


def annotate(anno_file, chromo, pos):
    anno_file = dat.GzipFile(anno_file, 'r')
    anno = pd.read_table(anno_file, header=None, usecols=[0, 1, 2],
                         dtype={0: 'str', 1: 'int32', 2: 'int32'})
    anno_file.close()
    anno.columns = ['chromo', 'start', 'end']
    anno.chromo = anno.chromo.str.upper().str.replace('CHR', '')
    anno = anno.loc[anno.chromo == chromo]
    anno.sort_values('start', inplace=True)
    start, end = an.join_overlapping(anno.start.values, anno.end.values)
    anno = np.array(an.is_in(pos, start, end), dtype='int8')
    return anno


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
            description='Creates DeepCpG data for training and testing.')

        # I/O
        p.add_argument(
            '--pos_file',
            help='File with positions of CpG sites that are to be predicted.'
            ' If missing, only CpG sites that are observed in at least one of'
            ' the given cells will be used.')
        p.add_argument(
            '--cpg_profiles',
            help='Input single-cell methylation profiles in dcpg or bedGraph'
            ' format that are to be imputed',
            nargs='+')
        p.add_argument(
            '--cpg_wlen',
            help='If provided, extract `cpg_wlen`//2 neighboring CpG sites',
            type=int)
        p.add_argument(
            '--cpg_cov',
            help='Minimum CpG coverage. Only use CpG sites for which the true'
            ' methylation state is known in at least that many cells.',
            type=int,
            default=1)
        p.add_argument(
            '--dna_files',
            help='Directory or FASTA files named "*.chromosome.`chromo`.fa*"'
            ' with the DNA sequences for chromosome `chromo`.',
            nargs='+')
        p.add_argument(
            '--dna_wlen',
            help='DNA window length',
            type=int,
            default=1001)
        p.add_argument(
            '--anno_files',
            help='Files with genomic annotations that are used as input'
            ' features. Currently ignored by `dcpg_train.py`.',
            nargs='+')
        p.add_argument(
            '-o', '--out_dir',
            help='Output directory',
            default='.')

        g = p.add_argument_group('output statistics')
        g.add_argument(
            '--cpg_stats',
            help='Per CpG statistics derived from single-cell profiles.'
            ' Required, e.g., for predicting mean methylation levels or'
            ' cell-to-cell variance.',
            nargs='+',
            choices=['mean', 'mode', 'var', 'cat_var', 'cat2_var', 'entropy',
                     'diff', 'cov'])
        g.add_argument(
            '--cpg_stats_cov',
            help='Minimum coverage for computing per CpG statistics',
            type=int,
            default=3)
        g.add_argument(
            '--win_stats',
            help='Window-based output statistics derived from single-cell'
            ' profiles. Required, e.g., for predicting mean methylation levels'
            ' or cell-to-cell variance.',
            nargs='+',
            choices=['mean', 'mode', 'var', 'cat_var', 'cat2_var', 'entropy',
                     'diff', 'cov'])
        g.add_argument(
            '--win_stats_wlen',
            help='Window lengths for computing statistics',
            type=int,
            nargs='+',
            default=[1001, 2001, 3001, 4001, 5001])

        g = p.add_argument_group('advanced arguments')
        g.add_argument(
            '--chromos',
            nargs='+',
            help='Chromosomes that are used')
        g.add_argument(
            '--nb_sample',
            type=int,
            help='Maximum number of samples')
        g.add_argument(
            '--nb_sample_chromo',
            type=int,
            help='Number of random samples from each chromosome')
        g.add_argument(
            '--chunk_size',
            type=int,
            default=32768,
            help='Maximum number of samples per output file. Should be'
            ' divisible by batch size.')
        g.add_argument(
            '--seed',
            help='Seed of random number generator',
            type=int,
            default=0)
        g.add_argument(
            '--verbose',
            help='More detailed log messages',
            action='store_true')
        g.add_argument(
            '--log_file',
            help='Write log messages to file')
        return p

    def main(self, name, opts):
        if opts.seed is not None:
            np.random.seed(opts.seed)

        logging.basicConfig(filename=opts.log_file,
                            format='%(levelname)s (%(asctime)s): %(message)s')
        log = logging.getLogger(name)
        if opts.verbose:
            log.setLevel(logging.DEBUG)
        else:
            log.setLevel(logging.INFO)
        log.debug(opts)

        # Check input arguments
        if not opts.cpg_profiles:
            if not (opts.pos_file or opts.dna_files):
                raise ValueError('Position table and DNA database expected!')

        if opts.dna_wlen and opts.dna_wlen % 2 == 0:
            raise '--dna_wlen must be odd!'
        if opts.cpg_wlen and opts.cpg_wlen % 2 != 0:
            raise '--cpg_wlen must be even!'

        # Parse functions for computing output statistics
        cpg_stats_meta = None
        win_stats_meta = None
        if opts.cpg_stats:
            cpg_stats_meta = get_stats_meta(opts.cpg_stats)
        if opts.win_stats:
            win_stats_meta = get_stats_meta(opts.win_stats)

        make_dir(opts.out_dir)
        outputs = OrderedDict()

        # Read single-cell profiles if provided
        if opts.cpg_profiles:
            log.info('Reading CpG profiles ...')
            outputs['cpg'] = read_cpg_profiles(
                opts.cpg_profiles,
                chromos=opts.chromos,
                nb_sample=opts.nb_sample,
                nb_sample_chromo=opts.nb_sample_chromo,
                log=log.info)

        # Create table with unique positions
        if opts.pos_file:
            # Read positions from file
            log.info('Reading position table ...')
            pos_table = pd.read_table(opts.pos_file, usecols=[0, 1],
                                      dtype={0: str, 1: np.int32},
                                      header=None, comment='#')
            pos_table.columns = ['chromo', 'pos']
            pos_table['chromo'] = dat.format_chromo(pos_table['chromo'])
            pos_table = prepro_pos_table(pos_table)
        else:
            # Extract positions from profiles
            pos_tables = []
            for cpg_table in list(outputs['cpg'].values()):
                pos_tables.append(cpg_table[['chromo', 'pos']])
            pos_table = prepro_pos_table(pos_tables)

        if opts.chromos:
            pos_table = pos_table.loc[pos_table.chromo.isin(opts.chromos)]
        if opts.nb_sample_chromo:
            pos_table = dat.sample_from_chromo(pos_table, opts.nb_sample_chromo)
        if opts.nb_sample:
            pos_table = pos_table.iloc[:opts.nb_sample]

        log.info('%d samples' % len(pos_table))

        make_dir(opts.out_dir)

        # Iterate over chromosomes
        # ------------------------
        for chromo in pos_table.chromo.unique():
            log.info('-' * 80)
            log.info('Chromosome %s ...' % (chromo))
            idx = pos_table.chromo == chromo
            chromo_pos = pos_table.loc[idx].pos.values
            chromo_outputs = OrderedDict()

            if 'cpg' in outputs:
                # Concatenate CpG tables into single nb_site x nb_output matrix
                chromo_outputs['cpg'] = map_cpg_tables(outputs['cpg'],
                                                       chromo, chromo_pos)
                chromo_outputs['cpg_mat'] = np.vstack(
                    list(chromo_outputs['cpg'].values())).T
                assert len(chromo_outputs['cpg_mat']) == len(chromo_pos)

            if 'cpg_mat' in chromo_outputs and opts.cpg_cov:
                cov = np.sum(chromo_outputs['cpg_mat'] != dat.CPG_NAN, axis=1)
                assert np.all(cov >= 1)
                idx = cov >= opts.cpg_cov
                tmp = '%s sites matched minimum coverage filter'
                tmp %= format_out_of(idx.sum(), len(idx))
                log.info(tmp)
                if idx.sum() == 0:
                    continue

                chromo_pos = chromo_pos[idx]
                chromo_outputs = select_dict(chromo_outputs, idx)

            # Read DNA of chromosome
            chromo_dna = None
            if opts.dna_files:
                chromo_dna = fasta.read_chromo(opts.dna_files, chromo)

            annos = None
            if opts.anno_files:
                log.info('Annotating CpG sites ...')
                annos = dict()
                for anno_file in opts.anno_files:
                    name = split_ext(anno_file)
                    annos[name] = annotate(anno_file, chromo, chromo_pos)

            # Iterate over chunks
            # -------------------
            nb_chunk = int(np.ceil(len(chromo_pos) / opts.chunk_size))
            for chunk in range(nb_chunk):
                log.info('Chunk \t%d / %d' % (chunk + 1, nb_chunk))
                chunk_start = chunk * opts.chunk_size
                chunk_end = min(len(chromo_pos), chunk_start + opts.chunk_size)
                chunk_idx = slice(chunk_start, chunk_end)
                chunk_pos = chromo_pos[chunk_idx]

                chunk_outputs = select_dict(chromo_outputs, chunk_idx)

                filename = 'c%s_%06d-%06d.h5' % (chromo, chunk_start, chunk_end)
                filename = os.path.join(opts.out_dir, filename)
                chunk_file = h5.File(filename, 'w')

                # Write positions
                chunk_file.create_dataset('chromo', shape=(len(chunk_pos),),
                                          dtype='S2')
                chunk_file['chromo'][:] = chromo.encode()
                chunk_file.create_dataset('pos', data=chunk_pos, dtype=np.int32)

                if len(chunk_outputs):
                    out_group = chunk_file.create_group('outputs')

                # Write cpg profiles
                if 'cpg' in chunk_outputs:
                    for name, value in six.iteritems(chunk_outputs['cpg']):
                        assert len(value) == len(chunk_pos)
                        # Round continuous values
                        out_group.create_dataset('cpg/%s' % name,
                                                 data=value.round(),
                                                 dtype=np.int8,
                                                 compression='gzip')
                    # Compute and write statistics
                    if cpg_stats_meta is not None:
                        log.info('Computing per CpG statistics ...')
                        cpg_mat = np.ma.masked_values(chunk_outputs['cpg_mat'],
                                                      dat.CPG_NAN)
                        mask = np.sum(~cpg_mat.mask, axis=1)
                        mask = mask < opts.cpg_stats_cov
                        for name, fun in six.iteritems(cpg_stats_meta):
                            stat = fun[0](cpg_mat).data.astype(fun[1])
                            stat[mask] = dat.CPG_NAN
                            assert len(stat) == len(chunk_pos)
                            out_group.create_dataset('cpg_stats/%s' % name,
                                                     data=stat,
                                                     dtype=fun[1],
                                                     compression='gzip')

                # Write input features
                in_group = chunk_file.create_group('inputs')

                # DNA windows
                if chromo_dna:
                    log.info('Extracting DNA sequence windows ...')
                    dna_wins = extract_seq_windows(chromo_dna, pos=chunk_pos,
                                                   wlen=opts.dna_wlen)
                    assert len(dna_wins) == len(chunk_pos)
                    in_group.create_dataset('dna', data=dna_wins, dtype=np.int8,
                                            compression='gzip')

                # CpG neighbors
                if opts.cpg_wlen:
                    log.info('Extracting CpG neighbors ...')
                    cpg_ext = fext.KnnCpgFeatureExtractor(opts.cpg_wlen // 2)
                    context_group = in_group.create_group('cpg')
                    # outputs['cpg'], since neighboring CpG sites might lie
                    # outside chunk borders and un-mapped values are needed
                    for name, cpg_table in six.iteritems(outputs['cpg']):
                        cpg_table = cpg_table.loc[cpg_table.chromo == chromo]
                        state, dist = cpg_ext.extract(chunk_pos,
                                                      cpg_table.pos.values,
                                                      cpg_table.value.values)
                        nan = np.isnan(state)
                        state[nan] = dat.CPG_NAN
                        dist[nan] = dat.CPG_NAN
                        # States can be binary (np.int8) or continuous
                        # (np.float32).
                        state = state.astype(cpg_table.value.dtype, copy=False)
                        dist = dist.astype(np.float32, copy=False)

                        assert len(state) == len(chunk_pos)
                        assert len(dist) == len(chunk_pos)
                        assert np.all((dist > 0) | (dist == dat.CPG_NAN))

                        group = context_group.create_group(name)
                        group.create_dataset('state', data=state,
                                             compression='gzip')
                        group.create_dataset('dist', data=dist,
                                             compression='gzip')

                if win_stats_meta is not None and opts.cpg_wlen:
                    log.info('Computing window-based statistics ...')
                    states = []
                    dists = []
                    cpg_states = []
                    cpg_group = out_group['cpg']
                    context_group = in_group['cpg']
                    for output_name in six.iterkeys(cpg_group):
                        state = context_group[output_name]['state'].value
                        states.append(np.expand_dims(state, 2))
                        dist = context_group[output_name]['dist'].value
                        dists.append(np.expand_dims(dist, 2))
                        cpg_states.append(cpg_group[output_name].value)
                    # samples x outputs x cpg_wlen
                    states = np.swapaxes(np.concatenate(states, axis=2), 1, 2)
                    dists = np.swapaxes(np.concatenate(dists, axis=2), 1, 2)
                    cpg_states = np.expand_dims(np.vstack(cpg_states).T, 2)
                    cpg_dists = np.zeros_like(cpg_states)
                    states = np.concatenate([states, cpg_states], axis=2)
                    dists = np.concatenate([dists, cpg_dists], axis=2)

                    for wlen in opts.win_stats_wlen:
                        idx = (states == dat.CPG_NAN) | (dists > wlen // 2)
                        states_wlen = np.ma.masked_array(states, idx)
                        group = out_group.create_group('win_stats/%d' % wlen)
                        for name, fun in six.iteritems(win_stats_meta):
                            stat = fun[0](states_wlen)
                            if hasattr(stat, 'mask'):
                                idx = stat.mask
                                stat = stat.data
                                if np.sum(idx):
                                    stat[idx] = dat.CPG_NAN
                            group.create_dataset(name, data=stat, dtype=fun[1],
                                                 compression='gzip')

                if annos:
                    log.info('Adding annotations ...')
                    group = in_group.create_group('annos')
                    for name, anno in six.iteritems(annos):
                        group.create_dataset(name, data=anno[chunk_idx],
                                             dtype='int8',
                                             compression='gzip')

                chunk_file.close()

        log.info('Done!')
        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
