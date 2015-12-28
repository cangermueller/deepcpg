#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import numpy as np
import pandas as pd
import h5py as h5


def gc_content(seq):
    return (seq.count('G') + seq.count('C')) / len(seq)


def cg_obs_exp(seq):
    f_cg = seq.count('CG') / len(seq)
    f_c = seq.count('C') / len(seq)
    f_g = seq.count('G') / len(seq)
    return np.log2(f_cg / (f_c * f_g))


def cg_per(seq):
    return seq.count('CG') / len(seq)


def c_per(seq):
    return seq.count('C') / len(seq)


def g_per(seq):
    return seq.count('G') / len(seq)


def t_per(seq):
    return seq.count('T') / len(seq)


def a_per(seq):
    return seq.count('A') / len(seq)


def adjust_pos(p, seq, target='CG'):
    for i in [0, -1, 1]:
        if seq[(p + i):(p + i + 2)] == target:
            return p + i
    return None


def extract_windows(seq, pos, wlen, callbacks, extend_win=False, seq_index=1):
    delta = wlen // 2
    n = pos.shape[0]
    seq = seq.upper()

    for i in range(n):
        p = pos[i] - seq_index
        q = adjust_pos(p, seq)
        assert q is not None, 'No CpG site!'
        win = seq[max(0, p - delta): min(len(seq), p + delta + 1)]
        if extend_win and len(win) < wlen:
            win = max(0, delta - p) * 'N' + win
            win += max(0, p + delta + 1 - len(seq)) * 'N'
            assert len(win) == wlen
        for callback in callbacks:
            callback(i, win)


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
            description='Extracts sequence windows over positions')
        p.add_argument(
            'seq_file',
            help='HDF path to chromosome sequences')
        p.add_argument(
            'pos_file',
            help='Position file with pos and chromo column')
        p.add_argument(
            '-o', '--out_file',
            help='Output HDF path')
        p.add_argument(
            '--wlen',
            help='Length of sequence window at positions',
            default=101,
            type=int)
        p.add_argument(
            '--stats',
            help='Statistics',
            nargs='+',
            default=['gc_content', 'cg_obs_exp'])
        p.add_argument(
            '--chromos',
            help='Only apply to these chromosome',
            nargs='+')
        p.add_argument(
            '--replace',
            help='Replace existing dataset',
            action='store_true')
        p.add_argument(
            '--verbose', help='More detailed log messages', action='store_true')
        p.add_argument(
            '--log_file', help='Write log messages to file')
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

        if opts.wlen % 2 == 0:
            raise ValueError('Window length must be odd!')

        stats = opts.stats
        stats_fun = [globals()[x] for x in stats]

        log.info('Read positions ...')
        pos = pd.read_table(opts.pos_file, comment='#',
                            header=None, dtype={0: 'str'})
        pos.columns = ['chromo', 'pos']
        if opts.chromos is not None:
            pos = pos.loc[pos.chromo.isin(opts.chromos)]
        pos.sort_values(['chromo', 'pos'], inplace=True)
        pos = pd.Series(pos.pos.values, index=pos.chromo.values)

        in_file = h5.File(opts.seq_file, 'r')
        out_file = h5.File(opts.out_file, 'a')
        for chromo in sorted(np.unique(pos.index.values)):
            log.info('Chromosome %s' % (chromo))
            cpos = pos.loc[chromo].values
            cseq = in_file['/%s' % (chromo)].value
            assert cpos[-1] < len(cseq), 'Invalid position'
            nb_seq = len(cpos)
            stats_out = []
            for s in range(len(stats)):
                t = np.empty(nb_seq, dtype='float32')
                stats_out.append(t)

            def extract_stats(i, seq):
                if i == 0 or i == nb_seq or i % (nb_seq // 20) == 0:
                    print('%.1f%%' % ((i + 1) / nb_seq * 100))
                for s in range(len(stats)):
                    stats_out[s][i] = stats_fun[s](seq)

            extract_windows(cseq, cpos, opts.wlen, [extract_stats])

            def write(name, data):
                p = '/%s/%s' % (chromo, name)
                if p in out_file:
                    if opts.replace:
                        del out_file[p]
                    else:
                        return
                out_file.create_dataset(p, data=data)

            write('pos', cpos)
            for s, stat in enumerate(stats):
                write(stat, stats_out[s])
        in_file.close()
        out_file.close()

        log.info('Done!')
        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
