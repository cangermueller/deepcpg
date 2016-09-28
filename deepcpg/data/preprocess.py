#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt

import h5py as h5
import numpy as np
import pandas as pd

from deepcpg.data import dna
from deepcpg.data import fasta
from deepcpg.data import io
from deepcpg import feature_extractor as fext


CPG_NAN=-1

# TODO:
# * Update comments
# * Check asserts
# * logging
# * check with missing args
def prepro_pos_table(pos_table):
    table = pos_table.groupby('chromo')
    table = table.apply(lambda df: pd.DataFrame({'pos': np.unique(df['pos'])}))
    table.reset_index(inplace=True)
    table = table[['chromo', 'pos']]
    table.sort_values(['chromo', 'pos'], inplace=True)
    return table


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
    if cpg_sites:
        assert np.all(seq_wins[:, delta] == 3)
        assert np.all(seq_wins[:, delta + 1] == 2)
    return seq_wins


def map_values(values, pos, target_pos, dtype=None, nan=CPG_NAN):
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
            type=int,
            help='Filter sites by CpG coverage')
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
            default='.',
            help='Output directory')
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

        pos_table = None
        if opts.pos_file:
            log.info('Read position table ...')
            pos_table = pd.read_table(opts.pos_file, usecols=[0, 1],
                                      dtype={0: str, 1: np.int32},
                                      header=None, comment='#')
            pos_table.columns = ['chromo', 'pos']
            if opts.chromos:
                pos_table = pos_table.loc[pos_table.chromo.isin(opts.chromos)]
            pos_table = prepro_pos_table(pos_table)

        cpg_tables = []
        target_names = []
        if opts.cpg_files:
            log.info('Read CpG files ...')
            for cpg_file in opts.cpg_files:
                cpg_tables.append(io.read_cpg_table(cpg_file, chromos=opts.chromos))
                target_names.append(pt.splitext(pt.basename(cpg_file))[0])
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
            chromo_cpgs = []
            for cpg_table in cpg_tables:
                cpg_table = cpg_table.loc[cpg_table.chromo == chromo]
                chromo_cpgs.append(map_values(cpg_table.value.values,
                                              cpg_table.pos.values,
                                              chromo_pos,
                                              dtype=np.int8))

            if opts.min_cpg_cov:
                idx = np.hstack([x.reshape(-1, 1) for x in chromo_cpgs])
                idx = np.sum(idx != CPG_NAN, axis=1) >= opts.min_cpg_cov
                tmp = '%s sites matched minimum coverage filter'
                tmp %= format_out_of(idx.sum(), len(idx))
                log.info(tmp)
                chromo_pos = chromo_pos[idx]
                for i in range(len(chromo_cpgs)):
                    chromo_cpgs[i] = chromo_cpgs[i][idx]

            nb_chunk = int(np.ceil(len(chromo_pos) / opts.chunk_size))

            chromo_dna = None
            if opts.dna_wlen:
                chromo_dna = fasta.read_chromo(opts.dna_db, chromo)

            for chunk in range(nb_chunk):
                log.info('Chunk \t%d / %d' % (chunk + 1, nb_chunk))
                chunk_start = chunk * opts.chunk_size
                chunk_end = min(len(chromo_pos), chunk_start + opts.chunk_size)
                chunk_pos = chromo_pos[chunk_start:chunk_end]
                path = 'c%s_%06d-%06d.h5' % (chromo, chunk_start, chunk_end)
                path = pt.join(opts.out_dir, path)
                chunk_file = h5.File(path, 'w')

                chunk_file.create_dataset('chromo', shape=(len(chunk_pos),), dtype='S2')
                chunk_file['chromo'][:] = chromo.encode()
                chunk_file.create_dataset('pos', data=chunk_pos, dtype=np.int32)

                if chromo_cpgs:
                    out_group = chunk_file.create_group('outputs')
                    for i, chromo_cpg in enumerate(chromo_cpgs):
                        name = 'cpg/%s' % target_names[i]
                        out_group.create_dataset(name,
                                                 data=chromo_cpg, dtype=np.int8,
                                                 compression='gzip')

                in_group = chunk_file.create_group('inputs')

                if chromo_dna:
                    log.info('Extract DNA sequence windows ...')
                    dna_wins = extract_seq_windows(chromo_dna, pos=chunk_pos, wlen=opts.dna_wlen)
                    in_group.create_dataset('dna', data=dna_wins, dtype=np.int8, compression='gzip')

                if opts.cpg_wlen:
                    log.info('Extract CpG neighbors ...')
                    cpg_ext = fext.KnnCpgFeatureExtractor(opts.cpg_wlen // 2)
                    context_group = in_group.create_group('cpg_context')
                    for target_name, cpg_table in zip(target_names, cpg_tables):
                        cpg_table = cpg_table.loc[cpg_table.chromo == chromo]
                        knn_state, knn_dist = cpg_ext.extract(chunk_pos, cpg_table.pos.values, cpg_table.value.values)
                        nan = np.isnan(knn_state)
                        knn_state = knn_state.astype(np.int8, copy=False)
                        knn_state[nan] = CPG_NAN
                        knn_dist = knn_dist.astype(np.float32, copy=False)
                        knn_dist[nan] = CPG_NAN

                        group = context_group.create_group(target_name)
                        group.create_dataset('state', data=knn_state, compression='gzip')
                        group.create_dataset('dist', data=knn_dist, compression='gzip')


                chunk_file.close()

        log.info('Done!')
        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
