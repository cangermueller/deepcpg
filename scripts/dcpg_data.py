#!/usr/bin/env python

import os
import re
import sys

import argparse
import logging
import h5py as h5
import numpy as np
import pandas as pd

from deepcpg import data as dat
from deepcpg.data import dna
from deepcpg.data import fasta
from deepcpg.data import feature_extractor as fext
from deepcpg.utils import EPS


# TODO:
# * Check asserts
# * check with missing args

def prepro_pos_table(pos_table):
    table = pos_table.groupby('chromo')
    table = table.apply(lambda df: pd.DataFrame({'pos': np.unique(df['pos'])}))
    table.reset_index(inplace=True)
    table = table[['chromo', 'pos']]
    table.sort_values(['chromo', 'pos'], inplace=True)
    return table


def output_name_from_filename(filename):
    name = os.path.splitext(os.path.basename(filename))[0]
    match = re.match(r'([^.]+)\.?', name)
    assert match
    name = match.group(1)
    return name


def extract_seq_windows(seq, pos, wlen, seq_index=1, cpg_sites=True):
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
        seq_wins[i] = dna.char2int(win)
    # Randomly choose missing nucleotides
    idx = seq_wins == dna.CHAR_TO_INT['N']
    seq_wins[idx] = np.random.randint(0, 4, idx.sum())
    assert seq_wins.max() < 4
    if cpg_sites:
        assert np.all(seq_wins[:, delta] == 3)
        assert np.all(seq_wins[:, delta + 1] == 2)
    return seq_wins


def map_values(values, pos, target_pos, dtype=None, nan=dat.CPG_NAN):
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


def format_out_of(out, of):
    return '%d / %d (%.1f%%)' % (out, of, out / of * 100)


def mean(x, axis=1):
    return np.mean(x, axis)


def var(x, axis=1):
    return x.var(axis=1)


def entropy(x, axis=1):
    p1 = x.mean(axis=axis)
    p1 = np.minimum(1 - EPS, np.maximum(EPS, p1))
    p0 = 1 - p1
    return -(p1 * np.log(p1) + p0 * np.log(p0))


def diff(x, axis=1):
    return np.array(x.min(axis=axis) != x.max(axis=axis), dtype=np.int8)


def disp(x, axis=1):
    mean = x.mean(axis=1)
    return x.var(axis=1) - mean * (1 - mean)


def output_stats_by_name(names):
    funs = dict()
    for name in names:
        if name == 'mean':
            fun = (mean, np.float32)
        elif name == 'var':
            fun = (var, np.float32)
        elif name == 'entropy':
            fun = (entropy, np.float32)
        elif name == 'diff':
            fun = (mean, np.int8)
        elif name == 'disp':
            fun = (disp, np.float32)
        else:
            raise ValueError('Invalid statistic "%s"!' % name)
        funs[name] = fun
    return funs


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
            description='Preprocesses data for training and testing.')
        p.add_argument(
            '--dna_db',
            help='DNA database file')
        p.add_argument(
            '--dna_wlen',
            type=int,
            help='DNA window length')
        p.add_argument(
            '--cpg_files',
            nargs='+',
            help='CpG BED files')
        p.add_argument(
            '--cpg_wlen',
            type=int,
            help='CpG window length')
        p.add_argument(
            '--pos_file',
            help='Position file')
        p.add_argument(
            '--min_cpg_cov',
            type=float,
            help='Filter sites by CpG coverage. Number of observations per '
                 'site, or percentage if smaller than 1.')
        p.add_argument(
            '--output_stats',
            help='Per CpG statistics to be computed as additional outputs',
            nargs='+',
            default=['cov', 'var', 'entropy', 'diff', 'disp'])
        p.add_argument(
            '--chromos',
            nargs='+',
            help='Filter data by chromosomes')
        p.add_argument(
            '--nb_sample',
            type=int,
            help='Maximum number of samples')
        p.add_argument(
            '--chunk_size',
            type=int,
            default=100000,
            help='Chunk size')
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

        if opts.dna_wlen and opts.dna_wlen % 2 == 0:
            raise '--dna_wlen must be odd!'
        if opts.cpg_wlen and opts.cpg_wlen % 2 != 0:
            raise '--cpg_wlen must be even!'

        output_stats = output_stats_by_name(opts.output_stats)

        pos_table = None
        if opts.pos_file:
            log.info('Reading position table ...')
            pos_table = pd.read_table(opts.pos_file, usecols=[0, 1],
                                      dtype={0: str, 1: np.int32},
                                      header=None, comment='#')
            pos_table.columns = ['chromo', 'pos']
            if opts.chromos:
                pos_table = pos_table.loc[pos_table.chromo.isin(opts.chromos)]
            pos_table = prepro_pos_table(pos_table)

        cpg_tables = []
        output_names = []
        if opts.cpg_files:
            log.info('Reading CpG files ...')
            for cpg_file in opts.cpg_files:
                _cpg_file = dat.GzipFile(cpg_file, 'r')
                tmp = dat.read_cpg_table(_cpg_file,
                                         chromos=opts.chromos,
                                         nrows=opts.nb_sample)
                cpg_tables.append(tmp)
                _cpg_file.close()
                output_names.append(output_name_from_filename(cpg_file))
            if pos_table is None:
                pos_table = []
                for cpg_table in cpg_tables:
                    pos_table.append(cpg_table[['chromo', 'pos']])
                pos_table = pd.concat(pos_table)
                pos_table = prepro_pos_table(pos_table)

        if opts.chromos:
            pos_table = pos_table.loc[pos_table.chromo.isin(opts.chromos)]
        if opts.nb_sample:
            pos_table = pos_table.iloc[:opts.nb_sample]

        for chromo in pos_table.chromo.unique():
            log.info('-' * 80)
            log.info('Chromosome %s ...' % (chromo))
            chromo_pos = pos_table.loc[pos_table.chromo == chromo].pos.values

            if cpg_tables:
                chromo_cpgs = []
                for cpg_table in cpg_tables:
                    cpg_table = cpg_table.loc[cpg_table.chromo == chromo]
                    chromo_cpgs.append(map_values(cpg_table.value.values,
                                                  cpg_table.pos.values,
                                                  chromo_pos,
                                                  dtype=np.int8))
                chromo_cpgs = np.vstack(chromo_cpgs).T

                if opts.min_cpg_cov:
                    min_cpg_cov = opts.min_cpg_cov
                    if min_cpg_cov < 1:
                        # Convert percentage to absolute number
                        min_cpg_cov = int(len(chromo_cpgs) * min_cpg_cov)
                        min_cpg_cov = max(min_cpg_cov, 1)
                    idx = np.sum(chromo_cpgs != dat.CPG_NAN, axis=1)
                    idx = idx >= min_cpg_cov
                    tmp = '%s sites matched minimum coverage filter'
                    tmp %= format_out_of(idx.sum(), len(idx))
                    log.info(tmp)
                    chromo_pos = chromo_pos[idx]
                    chromo_cpgs = chromo_cpgs[idx]
            else:
                chromo_cpgs = None

            nb_chunk = int(np.ceil(len(chromo_pos) / opts.chunk_size))

            chromo_dna = None
            if opts.dna_wlen:
                chromo_dna = fasta.read_chromo(opts.dna_db, chromo)

            for chunk in range(nb_chunk):
                log.info('Chunk \t%d / %d' % (chunk + 1, nb_chunk))
                chunk_start = chunk * opts.chunk_size
                chunk_end = min(len(chromo_pos), chunk_start + opts.chunk_size)
                chunk_pos = chromo_pos[chunk_start:chunk_end]
                filename = 'c%s_%06d-%06d.h5' % (chromo, chunk_start, chunk_end)
                filename = os.path.join(opts.out_dir, filename)
                chunk_file = h5.File(filename, 'w')

                chunk_file.create_dataset('chromo', shape=(len(chunk_pos),),
                                          dtype='S2')
                chunk_file['chromo'][:] = chromo.encode()
                chunk_file.create_dataset('pos', data=chunk_pos, dtype=np.int32)

                if chromo_cpgs is not None:
                    chunk_cpgs = chromo_cpgs[chunk_start:chunk_end]

                    out_group = chunk_file.create_group('outputs')
                    for i, output_name in enumerate(output_names):
                        name = 'cpg/%s' % output_name
                        out_group.create_dataset(name,
                                                 data=chunk_cpgs[:, i],
                                                 dtype=np.int8,
                                                 compression='gzip')
                    if output_stats:
                        chromo_cpgs = np.ma.masked_values(chromo_cpgs,
                                                          dat.CPG_NAN)
                        for name, fun in output_stats.items():
                            stat = fun[0](chunk_cpgs)
                            out_group.create_dataset('stats/%s' % name,
                                                     data=stat,
                                                     dtype=fun[1],
                                                     compression='gzip')

                in_group = chunk_file.create_group('inputs')

                if chromo_dna:
                    log.info('Extract DNA sequence windows ...')
                    dna_wins = extract_seq_windows(chromo_dna, pos=chunk_pos,
                                                   wlen=opts.dna_wlen)
                    in_group.create_dataset('dna', data=dna_wins, dtype=np.int8,
                                            compression='gzip')

                if opts.cpg_wlen:
                    log.info('Extract CpG neighbors ...')
                    cpg_ext = fext.KnnCpgFeatureExtractor(opts.cpg_wlen // 2)
                    context_group = in_group.create_group('cpg')
                    for output_name, cpg_table in zip(output_names, cpg_tables):
                        cpg_table = cpg_table.loc[cpg_table.chromo == chromo]
                        tmp = cpg_ext.extract(chunk_pos,
                                              cpg_table.pos.values,
                                              cpg_table.value.values)
                        knn_state, knn_dist = tmp
                        nan = np.isnan(knn_state)
                        knn_state = knn_state.astype(np.int8, copy=False)
                        knn_state[nan] = dat.CPG_NAN
                        knn_dist = knn_dist.astype(np.float32, copy=False)
                        knn_dist[nan] = dat.CPG_NAN

                        group = context_group.create_group(output_name)
                        group.create_dataset('state', data=knn_state,
                                             compression='gzip')
                        group.create_dataset('dist', data=knn_dist,
                                             compression='gzip')

                chunk_file.close()

        log.info('Done!')
        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
