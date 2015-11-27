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


def approx_chunk_size(mem, nb_target, nb_knn=None, seq_len=None):
    s = nb_target
    if nb_knn:
        s += 2 * nb_target * nb_knn * 2
    if seq_len:
        s += seq_len * 4
    return mem // s


def approx_mem(chunk_size, nb_target, nb_knn=None, seq_len=None):
    mem = 0
    mem += chunk_size * nb_target
    if nb_knn:
        mem += chunk_size * 2 * nb_target * nb_knn * 2
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
            'data_files',
            help='HDF path to data files',
            nargs='+')
        p.add_argument(
            '--target_files',
            help='Data files to be considered as targets',
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
            help='Max # CpGs',
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
            '--max_samples',
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

        target_files = opts.target_files
        if opts.target_files is None:
            target_files = opts.data_files

        # Get parameters
        nb_target = len(target_files)

        # KNN
        nb_knn = opts.knn
        if nb_knn is None:
            f = h5.File(target_files[0])
            nb_knn = f[
                '%s/%s/knn' %
                (opts.knn_group, opts.chromos[0])].shape[1]
            f.close()

        # Sequence length
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
        ltargetsy = []
        for i, target_file in enumerate(target_files):
            lfiles.append(pt.splitext(pt.basename(target_file))[0])
            t = 'u%d' % (i)
            ltargets.append(t)
            ltargetsy.append('%s_y' % (t))
        labels = dict(files=lfiles, targets=ltargets)
        for k, v in labels.items():
            f['/labels/%s' % (k)] = [x.encode() for x in v]

        # Get positions
        chromos = opts.chromos
        pos = dict()
        chromos_len = dict()
        for chromo in chromos:
            cpos = read_pos_all(target_files, chromo,
                               max_samples=opts.max_samples)
            chromos_len[chromo] = len(cpos)
            pos[chromo] = cpos
        posc = np.hstack([pos[x] for x in chromos])

        # Write positions
        fp = f.create_group('pos')
        N = len(posc)

        fp.create_dataset('pos', shape=(N,), dtype='int32')
        fp['chromos'] = [x.encode() for x in chromos]
        fp['chromos_len'] = [chromos_len[x] for x in chromos]

        fd = f.create_group('data')

        chunk_out = opts.chunk_out
        if not chunk_out and opts.max_mem and nb_knn > 0:
            chunk_out = approx_chunk_size(opts.max_mem * 10**6, nb_target,
                                          nb_knn, seq_len)
        if chunk_out:
            log.info('Using chunk size of %d (%.2f MB)' % (
                chunk_out,
                approx_mem(chunk_out, nb_target, nb_knn, seq_len) / 10**6
            ))

        for t in ltargetsy:
            s = (N,)
            fd.create_dataset(t, shape=s, dtype='int8',
                              chunks=chunk_size(s, chunk_out))

        if nb_knn > 0:
            s = (N, 2, nb_target, nb_knn)
            fd.create_dataset('c_x',
                            shape=s,
                            chunks=chunk_size(s, chunk_out),
                            dtype='float16')

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
            s = idx
            e = idx + chromos_len[chromo]
            shuffle = np.arange(chromos_len[chromo])
            if opts.shuffle:
                np.random.shuffle(shuffle)

            fp['pos'][s:e] = cpos[shuffle.argsort()]

            log.info('Read target CpG sites')
            for target, target_file in zip(ltargetsy, target_files):
                t = read_cpg(target_file, chromo, cpos)
                fd[target][s:e] = t[shuffle.argsort()]

            if nb_knn > 0:
                log.info('Read KNN')
                for i in range(0, len(cpos), opts.chunk_in):
                    j = i + opts.chunk_in
                    t = read_knn_all(opts.data_files,
                                    chromo=chromo,
                                    pos=cpos[i:j],
                                    knn_group=opts.knn_group,
                                    knn=nb_knn)
                    j = i + t.shape[0]
                    k = shuffle[i:j]
                    fd['c_x'][list(s + np.sort(k))] = t[k.argsort()]

            if opts.seq_file is not None:
                log.info('Read seq')
                t = read_seq(opts.seq_file, chromo, cpos, seq_len=seq_len)
                for i in range(0, t.shape[0], opts.chunk_in):
                    j = i + opts.chunk_in
                    tt = encode_seqs(t[i:j])
                    j = i + tt.shape[0]
                    k = shuffle[i:j]
                    fd['s_x'][list(s + np.sort(k))] = tt[k.argsort()]

            idx = e

        assert np.all(fp['pos'].value > 0)
        if not opts.shuffle:
            i = 0
            for chromo in chromos:
                j = i + chromos_len[chromo]
                cpos = fp['pos'][i:j]
                assert np.all(cpos[:-1] < cpos[1:])
                i = j
        f.close()


        log.info('Done!')

        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
