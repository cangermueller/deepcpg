#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import h5py as h5
import numpy as np


def read_pos(path, chromo, max_samples=None):
    f = h5.File(path, 'r')
    p = f['/cpg/%s/pos' % (chromo)]
    if max_samples:
        p = p[:max_samples]
    else:
        p = p.value
    return p

def adjust_pos(y, p, q):
    yq = np.empty(len(q), dtype='int8')
    yq.fill(-1)
    t = np.in1d(q, p).nonzero()[0]
    yq[t] = y
    return yq

def read_cpg(path, chromo, pos=None):
    f = h5.File(path, 'r')
    p = f['/cpg/%s/pos' % (chromo)].value
    c = f['/cpg/%s/cpg' % (chromo)].value
    f.close()
    if pos is not None:
        c = adjust_pos(c, p, pos)
        p = pos
    return (p, c)

def read_knn(path, chromo, pos=None):
    f = h5.File(path, 'r')
    p = f['/knn_shared/%s/pos' % (chromo)].value
    k = f['/knn_shared/%s/knn' % (chromo)].value
    d = f['/knn_shared/%s/dist' % (chromo)].value
    f.close()
    if pos is not None:
        t = np.in1d(p, pos)
        p = p[t]
        assert np.all(p == pos)
        k = k[t]
        d = d[t]
        assert k.shape[0] == d.shape[0] == len(pos)
    return (p, k, d)

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

def read_seq(path, chromo, pos=None):
    f = h5.File(path, 'r')
    p = f['/%s/pos' % (chromo)].value
    s = f['/%s/seq' % (chromo)].value
    #  p = f['/%s/pos' % (chromo)][:100]
    #  s = f['/%s/seq' % (chromo)][:100]
    f.close()
    if pos is not None:
        t = np.in1d(p, pos)
        p = p[t]
        assert np.all(p == pos)
        s = s[t]
        assert s.shape[0] == len(pos)
    return (p, s)

def read_chromo(data_files, chromo, max_samples=None, seq_file=None):
    pos = [read_pos(x, chromo, max_samples) for x in data_files]
    t = set()
    for p in pos:
        t.update(p)
    pos = np.array(sorted(t))

    cpgs = [read_cpg(x, chromo, pos)[1] for x in data_files]

    knns = [read_knn(x, chromo, pos) for x in data_files]
    kc = [x[1] for x in knns]
    n = kc[0].shape[0] # samples
    m = kc[0].shape[1] # win_len
    t = len(knns) # targets
    c = 2 # 2 features (knn, dist)
    kc = np.hstack(kc).reshape(n, t, m).reshape(-1) # reshape knn
    kd = np.hstack([x[2] for x in knns]).reshape(n, t, m).reshape(-1) # reshape distance
    knns = np.vstack((kc, kd)).T
    knns = knns.reshape(n, t, m, c) # combined
    knns = knns.swapaxes(2, 3).swapaxes(1, 2)
    assert knns.shape == (n, c, t, m) # samples x 2 x targets x win_len

    r = [pos, cpgs, knns]
    if seq_file:
        seq = read_seq(seq_file, chromo, pos)[1]
        r.append(seq)
    return r

def read(data_files, chromos, max_samples=None, seq_file=None):
    d = []
    for chromo in chromos:
        d.append(read_chromo(data_files, chromo, max_samples, seq_file))
    r = dict()
    # Positions
    r['pos'] = np.hstack([x[0] for x in d])
    r['chromos'] = [x.encode() for x in chromos]
    r['chromos_len'] = [len(x[0]) for x in d]
    # Target CpG sites
    cpgs = [x[1] for x in d]
    r['label_units'] = []
    r['label_files'] = []
    for i in range(len(cpgs[0])):
        c = [cpgs[j][i] for j in range(len(cpgs))]
        c = np.hstack(c)
        t = 'u%d_y' % (i)
        r[t] = c
        r['label_units'].append(t)
        r['label_files'].append(pt.splitext(pt.basename(data_files[i]))[0])
    for k in ['label_units', 'label_files']:
        r[k] = [x.encode() for x in r[k]]
    # CpG
    r['c_x'] = np.vstack([x[2] for x in d])
    # Seq
    if seq_file:
        r['s_x'] = np.vstack([x[3] for x in d])
        r['s_x'] = encode_seqs(r['s_x'])
    return r


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

        log.info('Preprocess data')
        data = read(seq_file=opts.seq_file,
                    data_files=opts.data_files,
                    chromos=opts.chromos,
                    max_samples=opts.max_samples)

        log.info('Write data')
        f = h5.File(opts.out_file, 'w')
        for k, v in data.items():
            f.create_dataset(k, data=v)
        f.close()

        log.info('Done!')

        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
