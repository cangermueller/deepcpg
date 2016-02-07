#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import h5py as h5
import numpy as np

import predict.utils as ut

MAX_DIST = 10**6


def read_anno(annos_file, chromo, name, pos=None):
    f = h5.File(annos_file, 'r')
    d = {k: f[pt.join(chromo, name, k)].value for k in ['pos', 'annos']}
    f.close()
    if pos is not None:
        t = np.in1d(d['pos'], pos)
        for k in d.keys():
            d[k] = d[k][t]
        assert np.all(d['pos'] == pos)
    d['annos'][d['annos'] >= 0] = 1
    d['annos'][d['annos'] < 0] = 0
    d['annos'] = d['annos'].astype('bool')
    return d['pos'], d['annos']


def read_annos(annos_file, chromo, names, *args, **kwargs):
    pos = None
    annos = []
    for name in names:
        p, a = read_anno(annos_file, chromo, name, *args, **kwargs)
        if pos is None:
            pos = p
        else:
            assert np.all(pos == p)
        annos.append(a)
    annos = np.vstack(annos).T
    return pos, annos


def read_stat(stats_file, chromo, name, pos=None):
    f = h5.File(stats_file, 'r')
    g = f[chromo]
    d = {k: g[k].value for k in [name, 'pos']}
    f.close()
    if pos is not None:
        t = np.in1d(d['pos'], pos)
        for k in d.keys():
            d[k] = d[k][t]
        assert np.all(d['pos'] == pos)
    return d['pos'], d[name]


def encode_seqs(seqs, dim=4):
    """Special nucleotides will be encoded as [0, 0, 0, 0]."""
    n = seqs.shape[0]
    l = seqs.shape[1]
    #  t = seqs >= dim
    #  seqs[t] = np.random.randint(0, dim, t.sum())
    enc_seqs = np.zeros((n, l, dim), dtype='int8')
    for i in range(dim):
        t = seqs == i
        enc_seqs[t, i] = 1
    return enc_seqs


def read_seq(path, chromo, pos=None, seq_len=None):
    f = h5.File(path, 'r')
    p = f['/%s/pos' % (chromo)].value
    s = f['/%s/seq' % (chromo)]
    if seq_len is None:
        s = s.value
    else:
        assert seq_len % 2 == 1
        assert seq_len <= s.shape[1]
        c = s.shape[1] // 2
        d = seq_len // 2
        s = s[:, c-d:c+d+1]
        assert s.shape[1] == seq_len
    f.close()
    if pos is not None:
        t = np.in1d(p, pos)
        p = p[t]
        assert np.all(p == pos)
        s = s[t]
        assert s.shape[0] == len(pos)
    return s


def chunk_size(shape, chunk_size):
    if chunk_size:
        c = list(shape)
        c[0] = min(chunk_size, c[0])
        c = tuple(c)
        return c
    else:
        return None


def approx_chunk_size(max_mem, nb_unit=1, nb_knn=None, seq_len=None):
    s = nb_unit
    max_ = [max_mem // nb_unit]
    if nb_knn:
        t = 2 * nb_unit * nb_knn * 2
        max_.append(max_mem // t)
        s += t
    if seq_len:
        t = seq_len * 4
        max_.append(max_mem // t)
        s += t
    max_.append(max_mem // s)
    return np.min(max_)


def approx_mem(chunk_size, nb_unit=1, nb_knn=None, seq_len=None):
    mem = 0
    mem += chunk_size * nb_unit
    if nb_knn:
        mem += chunk_size * 2 * nb_unit * nb_knn * 2
    if seq_len:
        mem += chunk_size * seq_len * 4
    return mem


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
            description='Preprocess data')
        p.add_argument(
            'seq_file',
            help='Sequence file')
        p.add_argument(
            'stats_file',
            help='Statistics file')
        p.add_argument(
            '-o', '--out_file',
            help='Output file')
        p.add_argument(
            '--stats',
            help='Statistics to be used as output variables',
            nargs='+',
            default=['w3000_var'])
        p.add_argument(
            '--seq_len',
            help='Sequence length',
            type=int,
            default=501)
        p.add_argument(
            '--annos_file',
            help='Annotation file')
        p.add_argument(
            '--annos',
            help='Regex of annotations to be considered',
            nargs='+')
        p.add_argument(
            '--annos_op',
            help='Operation to combine annos',
            choices=['and', 'or'],
            default='or')
        p.add_argument(
            '--chromos',
            help='Chromosomes',
            nargs='+',
            default=['1'])
        p.add_argument(
            '--chunk_in',
            help='Input chunk size',
            type=int,
            default=10**7)
        p.add_argument(
            '--chunk_out',
            help='Output (HDF) chunk size',
            type=int)
        p.add_argument(
            '--max_mem',
            help='Maximum memory load -> will adapt chunk_out',
            type=int,
            default=13000)
        p.add_argument(
            '--nb_sample',
            help='Limit # samples',
            type=int)
        p.add_argument(
            '--shuffle',
            help='Shuffle sequences',
            action='store_true')
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

        # Sequence length
        if opts.seq_file is not None:
            seq_len = opts.seq_len
            if seq_len is None:
                f = h5.File(opts.seq_file)
                seq_len = f['/%s/seq' % opts.chromos[0]].shape[1]
                f.close()
        else:
            seq_len = None

        stats_file = h5.File(opts.stats_file, 'r')

        # Get positions
        chromos = opts.chromos
        pos = dict()
        for chromo in chromos:
            pos[chromo] = stats_file[chromo]['pos'].value

        # Filter positions by annotations
        if opts.annos_file is not None:
            log.info('Filter positions by annotations')
            f = h5.File(opts.annos_file)
            names = list(f[chromos[0]].keys())
            f.close()
            if opts.annos is not None:
                names = ut.filter_regex(names, opts.annos)
            for chromo in chromos:
                t, annos = read_annos(opts.annos_file, chromo, names,
                                      pos[chromo])
                if opts.annos_op == 'or':
                    annos = annos.any(axis=1)
                else:
                    annos = annos.all(axis=1)
                pos[chromo] = pos[chromo][annos]

        # Concatenate position vector
        chromos_len = dict()
        posc = []
        for chromo in pos.keys():
            if opts.nb_sample is not None:
                pos[chromo] = pos[chromo][:opts.nb_sample]
            chromos_len[chromo] = len(pos[chromo])
            posc.append(pos[chromo])
        posc = np.hstack(posc)

        # Write target labels
        out_file = h5.File(opts.out_file, 'w')
        labels = dict()
        labels['targets'] = ['t%d' % (x) for x in range(len(opts.stats))]
        labels['files'] = opts.stats
        g = out_file.create_group('labels')
        for k, v in labels.items():
            g[k] = [x.encode() for x in v]

        # Initialize datasets
        fp = out_file.create_group('pos')
        N = len(posc)
        log.info('%d samples' % (N))

        fp.create_dataset('pos', shape=(N,), dtype='int32')
        fp.create_dataset('chromo', shape=(N,), dtype='S2', compression='gzip')

        fd = out_file.create_group('data')

        chunk_out = opts.chunk_out
        if not chunk_out and opts.max_mem is not None:
            chunk_out = approx_chunk_size(max_mem=opts.max_mem * 10**6,
                                          seq_len=seq_len)
        if chunk_out:
            log.info('Using chunk size of %d (%.2f MB)' % (
                chunk_out,
                approx_mem(chunk_out, seq_len=seq_len) / 10**6
            ))

        # Initialize target vectors
        for t in labels['targets']:
            s = (N, 1)
            fd.create_dataset(t + '_y', shape=s, dtype='float32',
                              chunks=chunk_size(s, chunk_out))

        # Initialize sequence matrix
        if opts.seq_file:
            s = (N, seq_len, 4)
            fd.create_dataset('s_x',
                              shape=s,
                              chunks=chunk_size(s, chunk_out),
                              dtype='int8')

        # Write data
        idx = 0
        for chromo in chromos:
            log.info('Chromosome %s' % (chromo))
            cpos = pos[chromo]
            Nc = chromos_len[chromo]
            s = idx
            e = idx + Nc
            shuffle = np.arange(Nc)
            if opts.shuffle:
                assert opts.chunk_in >= len(shuffle)
                np.random.shuffle(shuffle)

            fp['pos'][s:e] = cpos[shuffle.argsort()]
            fp['chromo'][s:e] = chromo.encode()

            log.info('Write targets')
            for target, stat in zip(labels['targets'], labels['files']):
                t, d = read_stat(opts.stats_file, chromo, stat, cpos)
                fd[target + '_y'][s:e, 0] = d[shuffle.argsort()]

            if opts.seq_file is not None:
                log.info('Read seq')
                # Read integer sequence (not one-hot encoded)
                ds = read_seq(opts.seq_file, chromo, cpos, seq_len=seq_len)
                chunk = 0
                nb_chunk_in = int(np.ceil(ds.shape[0] / opts.chunk_in))
                # Encode sequence in chunks to reduce storage
                for i in range(0, ds.shape[0], opts.chunk_in):
                    chunk += 1
                    log.info('Write seq (%d/%d)' % (chunk, nb_chunk_in))
                    j = i + opts.chunk_in
                    d = encode_seqs(ds[i:j])
                    j = i + d.shape[0]
                    k = shuffle[i:j]
                    t = list(s + np.sort(k))
                    t = np.array(t)
                    assert t.min() == s + i
                    assert t.max() == s + j - 1
                    fd['s_x'][s+i:s+j] = d[k.argsort()]

            idx = e

        assert np.all(fp['pos'].value > 0)
        if not opts.shuffle:
            i = 0
            for chromo in chromos:
                j = i + chromos_len[chromo]
                cpos = fp['pos'][i:j]
                assert np.all(cpos[:-1] < cpos[1:])
                i = j

        out_file.close()
        stats_file.close()

        log.info('Done!')

        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
