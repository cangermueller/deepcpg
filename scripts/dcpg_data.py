#!/usr/bin/env python

from collections import OrderedDict
import os
import sys

import argparse
import logging
import h5py as h5
import numpy as np
import pandas as pd

from deepcpg import data as dat
from deepcpg.data import annotations as an
from deepcpg.data import dna
from deepcpg.data import fasta
from deepcpg.data import feature_extractor as fext
from deepcpg.utils import EPS


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
    return os.path.basename(filename).split(os.extsep)[0]


def read_cpg_profiles(filenames, *args, **kwargs):
    cpg_profiles = OrderedDict()
    for filename in filenames:
        cpg_file = dat.GzipFile(filename, 'r')
        output_name = split_ext(filename)
        cpg_profile = dat.read_cpg_table(cpg_file, sort=True, *args, **kwargs)
        cpg_profiles[output_name] = cpg_profile
        cpg_file.close()
    return cpg_profiles


def extract_seq_windows(seq, pos, wlen, seq_index=1, cpg_sites=True):
    """Extracts DNA sequence windows at positions.

    Parameters
    ----------
    seq: DNA sequence string
    pos: Array with positions at which windows are extracted
    wlen: Window length
    seq_index: Minimum positions. Set to 0 if positions in `pos` start at 0
        instead of 1
    cpg_sites: Check if positions in `pos` point to CpG sites
    """

    delta = wlen // 2
    nb_win = len(pos)
    seq = seq.upper()
    seq_wins = np.zeros((nb_win, wlen), dtype='int8')

    for i in range(nb_win):
        p = pos[i] - seq_index
        if cpg_sites and seq[p:p + 2] != 'CG':
            raise ValueError('No CpG at position %d!' % p)
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
    if cpg_sites:
        assert np.all(seq_wins[:, delta] == 3)
        assert np.all(seq_wins[:, delta + 1] == 2)
    return seq_wins


def map_values(values, pos, target_pos, dtype=None, nan=dat.CPG_NAN):
    """Maps `values` array at positions `pos` to `target_pos`."""
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
    chromo_pos.sort()
    mapped_tables = OrderedDict()
    for name, cpg_table in cpg_tables.items():
        cpg_table = cpg_table.loc[cpg_table.chromo == chromo]
        cpg_table = cpg_table.sort_values(['chromo', 'pos'])
        mapped_table = map_values(cpg_table.value.values,
                                  cpg_table.pos.values,
                                  chromo_pos)
        assert len(mapped_table) == len(chromo_pos)
        mapped_tables[name] = mapped_table
    return mapped_tables


def format_out_of(out, of):
    return '%d / %d (%.1f%%)' % (out, of, out / of * 100)


def mean(x, axis=1):
    return np.mean(x, axis)


def mode(x, axis=1):
    return x.mean(axis=axis).round().astype(np.int8)


def var(x, *args, **kwargs):
    m = mean(x, *args, **kwargs)
    return m * (1 - m)


def cat_var(x, nb_bin=3, *args, **kwargs):
    v = var(x, *args, **kwargs)
    bins = np.linspace(-EPS, 0.25, nb_bin + 1)
    cv = np.digitize(v, bins, right=True) - 1
    return np.ma.masked_array(cv, v.mask)


def cat2_var(*args, **kwargs):
    cv = cat_var(*args, **kwargs)
    cv[cv > 0] = 1
    return cv


def entropy(x, axis=1):
    p1 = x.mean(axis=axis)
    p1 = np.minimum(1 - EPS, np.maximum(EPS, p1))
    p0 = 1 - p1
    return -(p1 * np.log(p1) + p0 * np.log(p0))


def diff(x, axis=1):
    return x.min(axis=axis) != x.max(axis=axis).astype(np.int8)


def output_stats_meta_by_name(names):
    funs = dict()
    for name in names:
        if name == 'mean':
            fun = (mean, np.float32)
        elif name == 'mode':
            fun = (mode, np.int8)
        elif name == 'var':
            fun = (var, np.float32)
        elif name == 'cat_var':
            fun = (cat_var, np.int8)
        elif name == 'cat2_var':
            fun = (cat2_var, np.int8)
        elif name == 'entropy':
            fun = (entropy, np.float32)
        elif name == 'diff':
            fun = (diff, np.int8)
        else:
            raise ValueError('Invalid statistic "%s"!' % name)
        funs[name] = fun
    return funs


def select_dict(data, idx):
    data = data.copy()
    for key, value in data.items():
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
            description='Prepares data for training and testing.')
        p.add_argument(
            '--dna_db',
            help='DNA database file')
        p.add_argument(
            '--dna_wlen',
            help='DNA window length',
            type=int,
            default=501)
        p.add_argument(
            '--cpg_profiles',
            help='BED files with single-cell methylation profiles',
            nargs='+')
        p.add_argument(
            '--bulk_profiles',
            help='BED files with bulk methylation profiles',
            nargs='+')
        p.add_argument(
            '--pos_file',
            help='Position file')
        p.add_argument(
            '--anno_files',
            help='Annotation files',
            nargs='+')
        p.add_argument(
            '--cpg_wlen',
            help='CpG window length',
            type=int)
        p.add_argument(
            '--cpg_cov',
            help='Minimum CpG coverage.',
            type=int,
            default=1)
        p.add_argument(
            '--cpg_stats',
            help='Output statistics derived from single-cell profiles',
            nargs='+',
            choices=['mean', 'mode', 'var', 'cat_var', 'cat2_var', 'entropy',
                     'diff'])
        p.add_argument(
            '--cpg_stats_cov',
            help='Minimum CpG coverage for computing statistics',
            type=int,
            default=1)
        p.add_argument(
            '--chromos',
            nargs='+',
            help='Select chromosomes')
        p.add_argument(
            '--nb_sample',
            type=int,
            help='Maximum number of samples')
        p.add_argument(
            '--chunk_size',
            type=int,
            default=32768,
            help='Chunk size. Should be divisible by batch size')
        p.add_argument(
            '-o', '--out_dir',
            help='Output directory',
            default='.')
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

        # Check input arguments
        if not (opts.cpg_profiles or opts.bulk_profiles):
            if not (opts.pos_file or opts.dna_db):
                raise ValueError('Position table and DNA database expected!')

        if opts.dna_wlen and opts.dna_wlen % 2 == 0:
            raise '--dna_wlen must be odd!'
        if opts.cpg_wlen and opts.cpg_wlen % 2 != 0:
            raise '--cpg_wlen must be even!'

        # Parse functions for computing output statistics
        if opts.cpg_stats:
            cpg_stats_meta = output_stats_meta_by_name(opts.cpg_stats)
        else:
            cpg_stats_meta = None

        outputs = OrderedDict()

        # Read single-cell profiles if provided
        if opts.cpg_profiles:
            log.info('Reading single-cell profiles ...')
            outputs['cpg'] = read_cpg_profiles(opts.cpg_profiles,
                                               chromos=opts.chromos,
                                               nrows=opts.nb_sample)

        if opts.bulk_profiles:
            log.info('Reading bulk profiles ...')
            outputs['bulk'] = read_cpg_profiles(opts.bulk_profiles,
                                                chromos=opts.chromos,
                                                nrows=opts.nb_sample,
                                                round=False)

        # Create table with unique positions
        if opts.pos_file:
            # Read positions from file
            log.info('Reading position table ...')
            pos_table = pd.read_table(opts.pos_file, usecols=[0, 1],
                                      dtype={0: str, 1: np.int32},
                                      header=None, comment='#')
            pos_table.columns = ['chromo', 'pos']
            pos_table = prepro_pos_table(pos_table)
        else:
            # Extract positions from profiles
            pos_tables = []
            for cpg_table in list(outputs['cpg'].values()):
                pos_tables.append(cpg_table[['chromo', 'pos']])
            pos_table = prepro_pos_table(pos_tables)
        log.info('%d samples' % len(pos_table))

        if opts.chromos:
            pos_table = pos_table.loc[pos_table.chromo.isin(opts.chromos)]
        if opts.nb_sample:
            pos_table = pos_table.iloc[:opts.nb_sample]

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

            if 'bulk' in outputs:
                # Concatenate CpG tables into single nb_site x nb_output matrix
                chromo_outputs['bulk'] = map_cpg_tables(outputs['bulk'],
                                                        chromo, chromo_pos)

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
            if opts.dna_db:
                chromo_dna = fasta.read_chromo(opts.dna_db, chromo)

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
                    for name, value in chunk_outputs['cpg'].items():
                        assert len(value) == len(chunk_pos)
                        out_group.create_dataset('cpg/%s' % name,
                                                 data=value,
                                                 dtype=np.int8,
                                                 compression='gzip')
                    # Compute and write statistics
                    if cpg_stats_meta is not None:
                        cpg_mat = np.ma.masked_values(chunk_outputs['cpg_mat'],
                                                      dat.CPG_NAN)
                        mask = np.sum(~cpg_mat.mask, axis=1)
                        mask = mask < opts.cpg_stats_cov
                        for name, fun in cpg_stats_meta.items():
                            stat = fun[0](cpg_mat).data.astype(fun[1])
                            stat[mask] = dat.CPG_NAN
                            assert len(stat) == len(chunk_pos)
                            out_group.create_dataset('stats/%s' % name,
                                                     data=stat,
                                                     dtype=fun[1],
                                                     compression='gzip')

                # Write bulk profiles
                if 'bulk' in chunk_outputs:
                    for name, value in chunk_outputs['bulk'].items():
                        assert len(value) == len(chunk_pos)
                        out_group.create_dataset('bulk/%s' % name,
                                                 data=value,
                                                 dtype=np.float32,
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
                    for name, cpg_table in outputs['cpg'].items():
                        cpg_table = cpg_table.loc[cpg_table.chromo == chromo]
                        state, dist = cpg_ext.extract(chunk_pos,
                                                      cpg_table.pos.values,
                                                      cpg_table.value.values)
                        nan = np.isnan(state)
                        state[nan] = dat.CPG_NAN
                        dist[nan] = dat.CPG_NAN
                        state = state.astype(np.int8, copy=False)
                        dist = dist.astype(np.float32, copy=False)

                        assert len(state) == len(chunk_pos)
                        assert np.all((state == 0) | (state == 1) |
                                      (state == dat.CPG_NAN))
                        assert len(dist) == len(chunk_pos)
                        assert np.all((dist > 0) | (dist == dat.CPG_NAN))

                        group = context_group.create_group(name)
                        group.create_dataset('state', data=state,
                                             compression='gzip')
                        group.create_dataset('dist', data=dist,
                                             compression='gzip')

                if annos:
                    group = in_group.create_group('annos')
                    for name, anno in annos.items():
                        group.create_dataset(name, data=anno[chunk_idx],
                                             dtype='int8',
                                             compression='gzip')

                chunk_file.close()

        log.info('Done!')
        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
