#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import h5py as h5
import numpy as np
import gc

import predict.utils as ut
from predict.models.dnn.utils import MASK


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


def read_pos(path, chromo, nb_sample=None):
    f = h5.File(path, 'r')
    p = f['/cpg/%s/pos' % (chromo)]
    if nb_sample:
        p = p[:nb_sample]
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
    yq.fill(MASK)
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


def chunk_size(shape, chunk_size):
    if chunk_size:
        c = list(shape)
        c[0] = min(chunk_size, c[0])
        c = tuple(c)
        return c
    else:
        return None


def approx_chunk_size(max_mem, nb_target=1, seq_len=None,
                      nb_unit=None, nb_knn=None,
                      max_chunk_mem=4*10**9):
    # Maximum HDF chunk size is 4GB
    max_ = [max_chunk_mem]
    # Memory target vectors (float32)
    s = nb_target * 2
    max_.append(max_chunk_mem // s)
    # Memory knn array
    if nb_unit is not None:
        t = 2 * nb_unit * nb_knn * 2
        max_.append(max_chunk_mem // t)
        s += t
    # Memory sequence array
    if seq_len:
        t = seq_len * 4
        max_.append(max_chunk_mem // t)
        s += t
    max_.append(max_mem // s)
    return np.min(max_)


def approx_mem(chunk_size, nb_target=1, seq_len=None,
               nb_unit=None, nb_knn=None):
    mem = 0
    mem += chunk_size * nb_target * 2
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
            '--cpg_knn',
            help='CpG files to be used as knn',
            nargs='+')
        p.add_argument(
            '--cpg_targets',
            help='CpG files to be used as targets',
            nargs='+')
        p.add_argument(
            '--seq_file',
            help='HDF path to seq file')
        p.add_argument(
            '--stats_file',
            help='Statistics file')
        p.add_argument(
            '--stats_targets',
            help='Target statistics',
            nargs='+',
            default=['w3000_var'])
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
            '-o', '--out_file',
            help='Output file',
            default='./data.h5')
        p.add_argument(
            '--chromos',
            help='Chromosomes',
            nargs='+',
            default=['1'])
        p.add_argument(
            '--seq_len',
            help='Sequence length',
            type=int)
        p.add_argument(
            '--knn',
            help='Max # CpGs',
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

        if opts.cpg_knn is None and opts.seq_file is None:
            raise 'No input given'

        if opts.cpg_targets is None and opts.stats_file is None:
            raise 'No targets given!'

        # Get target positions
        pos = dict()
        chromos = opts.chromos
        if opts.stats_file is not None:
            # Statistics determine positions if both stat and CpG targets
            stats_file = h5.File(opts.stats_file, 'r')
            for chromo in chromos:
                pos[chromo] = stats_file[chromo]['pos'].value
        else:
            for chromo in chromos:
                pos[chromo] = read_pos_all(opts.cpg_targets, chromo,
                                           nb_sample=opts.nb_sample)
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
        N = len(posc)
        print('Number of samples: %d' % (N))

        # Initialize variables
        nb_unit = None
        seq_len = None
        nb_knn = None
        if opts.cpg_knn is not None:
            nb_unit = len(opts.cpg_knn)
            nb_knn = opts.knn
            if nb_knn is None:
                f = h5.File(opts.cpg_knn[0])
                nb_knn = f[
                    '%s/%s/knn' %
                    (opts.knn_group, opts.chromos[0])].shape[1]
                f.close()
        if opts.seq_file is not None:
            seq_len = opts.seq_len
            if seq_len is None:
                f = h5.File(opts.seq_file)
                seq_len = f['/%s/seq' % opts.chromos[0]].shape[1]
                f.close()

        target_ids = []
        target_names = []
        target_files = []
        if opts.cpg_targets is not None:
            for i, target in enumerate(opts.cpg_targets):
                target_ids.append('c%d' % (i))
                target_names.append(pt.splitext(pt.basename(target))[0])
                target_files.append(target)
        if opts.stats_file is not None:
            for i, target in enumerate(opts.stats_targets):
                target_ids.append('s%d' % (i))
                target_names.append(target)
                target_files.append(opts.stats_file)
        nb_target = len(target_names)
        print('Targets:')
        for target_id, target_name in zip(target_ids, target_names):
            print('%s: %s' % (target_id, target_name))

        log.info('Initialize data file')

        # Write labels
        out_file = h5.File(opts.out_file, 'w')
        out_file['/targets/id'] = [x.encode() for x in target_ids]
        out_file['/targets/name'] = [x.encode() for x in target_names]

        # Write positions
        fp = out_file.create_group('pos')
        fp.create_dataset('pos', shape=(N,), dtype='int32')
        fp.create_dataset('chromo', shape=(N,), dtype='S2', compression='gzip')

        # Initialize datasets
        fd = out_file.create_group('data')
        chunk_out = opts.chunk_out
        if not chunk_out and opts.max_mem:
            chunk_out = approx_chunk_size(opts.max_mem * 10**6, nb_target,
                                          seq_len, nb_unit, nb_knn)
        if chunk_out:
            t = approx_mem(chunk_out, nb_target, seq_len, nb_unit, nb_knn)
            t /= 10**6
            print('Chunk size: %d (%.2f MB)' % (chunk_out, t))

        for t in target_ids:
            s = (N, 1)
            if t.startswith('c'):
                dtype = 'int8'
            else:
                dtype = 'float32'
            fd.create_dataset('%s_y' % (t), shape=s, dtype=dtype,
                              chunks=chunk_size(s, chunk_out))

        if nb_knn is not None:
            s = (N, 2, nb_unit, nb_knn)
            fd.create_dataset('c_x', shape=s, chunks=chunk_size(s, chunk_out),
                              dtype='float16')

        if seq_len is not None:
            s = (N, seq_len, 4)
            fd.create_dataset('s_x', shape=s, chunks=chunk_size(s, chunk_out),
                              dtype='int8')

        # Write data
        log.info('Write data')
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
            for i in range(nb_target):
                target_id = target_ids[i]
                target_name = target_names[i]
                target_file = target_files[i]
                if target_id.startswith('s'):
                    t, d = read_stat(target_file, chromo, target_name, cpos)
                    assert np.all((d >= 0) & (d <= 1))
                else:
                    d = read_cpg(target_file, chromo, cpos)
                    if nb_target == 1:
                        assert np.all((d == 0) | (d == 1))
                    else:
                        assert np.all((d == 0) | (d == 1) | (d == MASK))
                fd['%s_y' % (target_id)][s:e, 0] = d[shuffle.argsort()]

            if nb_knn is not None:
                chunk = 0
                nb_chunk_in = int(np.ceil(len(cpos) / opts.chunk_in))
                for i in range(0, len(cpos), opts.chunk_in):
                    chunk += 1
                    log.info('Read KNN (%d/%d)' % (chunk, nb_chunk_in))
                    j = i + opts.chunk_in
                    d = read_knn_all(opts.cpg_knn, chromo=chromo,
                                     pos=cpos[i:j],
                                     knn_group=opts.knn_group,
                                     knn=nb_knn)
                    j = i + d.shape[0]
                    k = shuffle[i:j]
                    log.info('Write KNN (%d/%d)' % (chunk, nb_chunk_in))
                    t = list(s + np.sort(k))
                    t = np.array(t)
                    assert t.min() == s + i
                    assert t.max() == s + j - 1
                    fd['c_x'][s+i:s+j] = d[k.argsort()]

            if seq_len is not None:
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
        log.info('Done!')
        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
