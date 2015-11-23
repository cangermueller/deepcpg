#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import h5py as h5
import numpy as np
import gc


MAX_DIST = 10**6


def read_pos(path, chromo, max_samples=None):
    f = h5.File(path, 'r')
    p = f['/cpg/%s/pos' % (chromo)]
    if max_samples:
        p = p[:max_samples]
    else:
        p = p.value
    return p


def read_pos_all(data_files, *args, **kwargs):
    pos = [read_pos(x, *args, **kwargs) for x in data_files]
    t = set()
    for p in pos:
        t.update(p)
    pos = np.array(sorted(t))
    return pos


def adjust_pos(y, p, q):
    yq = np.empty(len(q), dtype='int8')
    yq.fill(-1)
    t = np.in1d(q, p).nonzero()[0]
    yq.flat[t] = y
    return yq


def read_cpg(path, chromo, pos=None):
    f = h5.File(path, 'r')
    p = f['/cpg/%s/pos' % (chromo)].value
    c = f['/cpg/%s/cpg' % (chromo)].value
    f.close()
    if pos is not None:
        c = adjust_pos(c, p, pos)
        p = pos
    c = c.astype('int8')
    return c


def read_knn(path, chromo, pos=None, what='knn', knn_group='knn_shared',
             knn=None):
    f = h5.File(path, 'r')
    g = f['/%s/%s' % (knn_group, chromo)]
    p = g['pos'].value
    d = g[what]
    if knn is None:
        d = d.value
    else:
        assert knn % 2 == 0
        assert knn <= d.shape[1]
        c = d.shape[1] // 2
        t = knn // 2
        d = d[:, c-t:c+t]
    f.close()
    if pos is not None:
        t = np.in1d(p, pos)
        d = d[t]
    return d


def read_knn_dist(max_dist=MAX_DIST, *args, **kwargs):
    d = read_knn(what='dist', *args, **kwargs)
    d = np.array(np.minimum(max_dist, d) / max_dist, dtype='float16')
    return d


def read_knn_all(paths, *args, **kwargs):
    d = [read_knn(x, *args, **kwargs) for x in paths]

    T = len(d)  # targets
    N = d[0].shape[0]  # samples
    M = d[0].shape[1]  # win_len
    C = 2  # features
    d = np.hstack(d).reshape(N, T, M).reshape(-1)

    t = [read_knn_dist(path=x, *args, **kwargs) for x in paths]
    t = np.hstack(t).reshape(N, T, M).reshape(-1)

    d = np.vstack((d, t)).T
    del t
    gc.collect()

    d = d.reshape(N, T, M, C)  # combined
    d = d.swapaxes(2, 3).swapaxes(1, 2)
    assert d.shape == (N, C, T, M)
    return d


def encode_seqs(seqs):
    n = seqs.shape[0]
    l = seqs.shape[1]
    enc_seqs = np.zeros((n, l, 4), dtype='float16')
    for i in range(n):
        seq = seqs[i]
        enc_seq = enc_seqs[i]
        special = seq == 4
        enc_seq[special, :] = 0.25
        enc_seq[~special, seq[~special]] = 1
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
            'data_files',
            help='HDF path to data files',
            nargs='+')
        p.add_argument(
            '--seq_file',
            help='HDF path to seq file')
        p.add_argument(
            '-o', '--out_file',
            help='Output file')
        p.add_argument(
            '--chromos',
            help='Chromosomes',
            nargs='+',
            default=['1'])
        p.add_argument(
            '--knn',
            help='Max # knn',
            type=int)
        p.add_argument(
            '--seq_len',
            help='Sequence length',
            type=int)
        p.add_argument(
            '--knn_group',
            help='Name of knn group in HDF file',
            default='knn_shared')
        p.add_argument(
            '--chunk_size',
            help='Max # records to read',
            type=int,
            default=10**7)
        p.add_argument(
            '--max_samples',
            help='Limit # samples',
            type=int)
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

        # Get KNN
        knn = opts.knn
        if knn is None:
            f = h5.File(opts.data_files[0])
            knn = f[
                '%s/%s/knn' %
                (opts.knn_group, opts.chromos[0])].shape[1]
            f.close()

        # Get sequence length
        if opts.seq_file is not None:
            seq_len = opts.seq_len
            if seq_len is None:
                f = h5.File(opts.seq_file)
                seq_len = f['/%s/seq' % opts.chromos[0]].shape[1]
                f.close()

        # Write target labels
        f = h5.File(opts.out_file, 'w')
        labels = dict()
        lfiles = []
        ltargets = []
        for i, data_file in enumerate(opts.data_files):
            lfiles.append(pt.splitext(pt.basename(data_file))[0])
            ltargets.append('u%d_y' % (i))
        labels = dict(files=lfiles, targets=ltargets)
        for k, v in labels.items():
            f['/labels/%s' % (k)] = [x.encode() for x in v]

        # Get positions
        chromos = opts.chromos
        pos = dict()
        chromos_len = dict()
        for chromo in chromos:
            cpos = read_pos_all(opts.data_files, chromo,
                               max_samples=opts.max_samples)
            chromos_len[chromo] = len(cpos)
            pos[chromo] = cpos
        posc = np.hstack([pos[x] for x in chromos])

        # Write positions
        g = f.create_group('pos')
        N = len(posc)
        g['pos'] = posc
        g['chromos'] = [x.encode() for x in chromos]
        g['chromos_len'] = [chromos_len[x] for x in chromos]

        fd = f.create_group('data')

        for t in ltargets:
            fd.create_dataset(t, shape=(N,), dtype='float16')

        fd.create_dataset('c_x',
                         shape=(N, 2, len(opts.data_files), knn),
                         dtype='float16')

        fd.create_dataset('s_x',
                         shape=(N, seq_len, 4),
                         dtype='float16')

        # Write data
        idx = 0
        for chromo in chromos:
            log.info('Chromosome %s' % (chromo))
            cpos = pos[chromo]
            start = idx
            end = idx + chromos_len[chromo]

            log.info('Read CpG sites')
            for target, data_file in zip(ltargets, opts.data_files):
                d = read_cpg(data_file, chromo, cpos)
                fd[target][start:end] = d

            log.info('Read KNN')
            for i in range(0, len(cpos), opts.chunk_size):
                j = i + opts.chunk_size
                t = read_knn_all(opts.data_files,
                                 chromo=chromo,
                                 pos=cpos[i:j],
                                 knn_group=opts.knn_group,
                                 knn=knn)
                fd['c_x'][start+i:start+i+t.shape[0]] = t

            if opts.seq_file is not None:
                log.info('Read seq')
                t = read_seq(opts.seq_file, chromo, cpos, seq_len=seq_len)
                for i in range(0, t.shape[0], opts.chunk_size):
                    j = i + opts.chunk_size
                    tt = encode_seqs(t[i:j])
                    fd['s_x'][start+i:start+i+tt.shape[0]] = tt
            idx = end

        f.close()

        log.info('Done!')

        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
