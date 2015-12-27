#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import numpy as np
import numpy.ma
import h5py as h5
import progressbar


def read_pos(path, chromo, max_samples=None):
    f = h5.File(path, 'r')
    p = f['/cpg/%s/pos' % (chromo)]
    if max_samples:
        p = p[:max_samples]
    else:
        p = p.value
    return p

def adjust_pos(y, p, q, mask=-1):
    yq = np.empty(len(q), dtype='int8')
    yq.fill(-1)
    t = np.in1d(q, p).nonzero()[0]
    yq[t] = y
    return yq

def read_cpg(path, chromo, pos=None, *args, **kwargs):
    f = h5.File(path, 'r')
    p = f['/cpg/%s/pos' % (chromo)].value
    c = f['/cpg/%s/cpg' % (chromo)].value
    f.close()
    if pos is not None:
        c = adjust_pos(c, p, pos, *args, **kwargs)
        p = pos
    return (p, c)

def read_chromo(data_files, chromo, max_samples=None):
    pos = [read_pos(x, chromo, max_samples) for x in data_files]
    t = set()
    for p in pos:
        t.update(p)
    pos = np.array(sorted(t))

    d = [read_cpg(x, chromo, pos, mask=-1)[1] for x in data_files]
    d = np.atleast_2d(np.vstack(d)).T
    d = np.ma.masked_values(d, -1)
    return (pos, d)

def cov(x):
    return np.sum(~x.mask, axis=1) / x.shape[1]

def var(x):
    return x.var(axis=1)

def entropy(x, mean=False):
    assert x.mask.ndim == 2
    n = np.sum(~x.mask, axis=1)
    p1 = np.sum(x == 1, axis=1) / n
    eps = 1e-8
    p1[p1 == 0] = eps
    p0 = np.sum(x == 0, axis=1) / n
    p0[p0 == 0] = eps
    e = -(p1 * np.log(p1 + 1e-8) + p0 * np.log(p0 + 1e-8))
    if mean:
        e = e.mean()
    assert np.all(e >= 0)
    return e

def win_entropy(x, *args, **kwargs):
    return entropy(x, mean=True)

def win_dist(x, p, c, *args, **kwargs):
    assert x.mask.ndim == 2
    s = [i for i in range(len(p)) if i != c]
    d = np.abs(p - p[c])[s]
    w = np.sum(~x.mask, axis=1)[s]
    return np.sum(d * w) / (w.sum() + 1e-8)

def win_cov(x, *args, **kwargs):
    assert x.mask.ndim == 2
    t = np.mean(np.sum(~x.mask, axis=1) / x.shape[1])
    assert t >= 0 and t <= 1
    return t

def win_var(x, *args, **kwargs):
    return x.mean(axis=0).var()

def rolling_apply(pos, x, delta, funs, y=None, callback=None):
    n = len(pos)
    if not isinstance(funs, list):
        funs = list(funs)
    if y is None:
        y = np.array((n, len(funs)), dtype='float32')

    l = 0
    r = 0
    for i in range(n):
        if callback is not None:
            callback(i)
        p = pos[i]
        while l < i and p - pos[l] > delta:
            l += 1
        while r < len(pos) - 1 and pos[r + 1] - p <= delta:
            r += 1
        xi = x[l:(r + 1)]
        pi = pos[l:(r + 1)]
        for j in range(len(funs)):
            y[i, j] = funs[j](xi, pi, i - l)
    return y


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
            'data_files',
            help='HDF path to data files',
            nargs='+')
        p.add_argument(
            '-o', '--out_file',
            help='Output HDF path')
        p.add_argument(
            '--stats',
            help='Statistics',
            nargs='+',
            default=['cov', 'var', 'entropy',
                     'win_cov', 'win_var', 'win_entropy', 'win_dist'])
        p.add_argument(
            '--wlen',
            help='Sliding window length',
            type=int,
            default=3000)
        p.add_argument(
            '--chromos',
            help='Only apply to these chromosome',
            nargs='+')
        p.add_argument(
            '--max_samples',
            help='Only consider that many samples',
            type=int)
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

        delta = opts.wlen // 2
        funs = list()
        win_funs = list()
        for stat in opts.stats:
            if stat.startswith('win'):
                win_funs.append((stat, globals()[stat]))
            else:
                funs.append((stat, globals()[stat]))

        chromos = opts.chromos
        if chromos is None:
            f = h5.File(opts.data_files[0])
            chromos = list(f['cpg'].keys())
            f.close()

        f = h5.File(opts.out_file, 'a')
        for chromo in chromos:
            log.info('Chromosome %s' % (chromo))
            pos, X = read_chromo(opts.data_files, chromo, opts.max_samples)
            t = pt.join(chromo, 'pos')
            if t in f:
                del f[t]
            f.create_dataset(t, data=pos)
            log.info('Per CpG statistics')
            for k, v in funs:
                y = v(X)
                t = pt.join(chromo, k)
                if t in f:
                    del f[t]
                f.create_dataset(t, data=y)
            log.info('Window-based statistics')
            n = len(pos)
            y = np.empty((n, len(win_funs)), dtype='float32')
            def prog(i):
                if i % 10000 == 0:
                    log.info('%4.1f%%' % (i / n * 100))
            y = rolling_apply(pos, X,
                              funs=[x[1] for x in win_funs],
                              delta=delta, y=y,
                              callback=prog)
            for i, x in enumerate([x[0] for x in win_funs]):
                t = pt.join(chromo, x)
                if t in f:
                    del f[t]
                f.create_dataset(t, data=y[:, i])
        f.close()

        log.info('Done!')
        return 0

if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
