#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import h5py as h5
import numpy as np


def read_knn(data_file, chromo, pos, knn, knn_group):
    f = h5.File(data_file)
    g = f[pt.join(knn_group, chromo)]
    p = g['pos'].value
    c = g['knn'].shape[1] // 2
    K = g['knn'][:, c - knn: c + knn]
    D = g['dist'][:, c - knn: c + knn]
    f.close()

    t = np.in1d(p, pos)
    p = p[t]
    assert np.all(p == pos)
    K = K[t]
    D = D[t]
    K = np.hstack((K, D))

    cols = ['knn_l%d' % (x + 1) for x in range(knn)]
    cols.extend(['knn_r%d' % (x + 1) for x in range(knn)])
    cols.extend(['knn_dl%d' % (x + 1) for x in range(knn)])
    cols.extend(['knn_dr%d' % (x + 1) for x in range(knn)])

    return (K, cols)


def read_annos(annos_file, chromo, name, pos=None, binary=True):
    f = h5.File(annos_file)
    d = {k: f[pt.join(chromo, name, k)].value for k in ['pos', 'annos']}
    f.close()
    if pos is not None:
        t = np.in1d(d['pos'], pos)
        for k in d.keys():
            d[k] = d[k][t]
        assert np.all(d['pos'] == pos)
    p, a = d['pos'], d['annos']
    if binary:
        a[a >= 0] = 1
        a[a < 0] = 0
        a = a.astype('int8')
    return (p, a)


def read_annos_matrix(annos_file, chromo, pos, annos=None, annos_excl=None):
    if annos is None:
        f = h5.File(annos_file)
        if annos is None:
            annos = sorted(list(f[chromo].keys()))
    if annos_excl is not None:
        annos = list(filter(lambda x: x not in annos_excl, annos))
    d = [read_annos(annos_file, chromo, x, pos) for x in annos]
    p = d[0][0]
    A = np.vstack([x[1] for x in d]).T
    assert len(p) == A.shape[0]
    return (A, annos)


def read_kmers(path, chromo, pos=None):
    f = h5.File(path)
    g = f[chromo]
    labels = [x.decode() for x in g['labels']]
    kmers = g['kmers'].value
    p = g['pos'].value
    f.close()
    if pos is not None:
        t = np.in1d(p, pos)
        kmers = kmers[t]
        p = p[t]
        assert np.all(p == pos)
    return (kmers, labels)


def read_chromo(data_file, chromo, knn, knn_group, annos_file=None,
                max_samples=None, annos_excl=None, kmers_file=None):
    f = h5.File(data_file)
    pos = f[pt.join('cpg', chromo, 'pos')]
    y = f[pt.join('cpg', chromo, 'cpg')]
    if max_samples is None:
        pos = pos.value
        y = y.value
    else:
        pos = pos[:max_samples]
        y = y[:max_samples]
    f.close()

    X, cols = read_knn(data_file, chromo, pos, knn=knn, knn_group=knn_group)

    if annos_file is not None:
        A, Acols = read_annos_matrix(annos_file, chromo, pos,
                                     annos_excl=annos_excl)
        X = np.hstack((X, A))
        cols.extend(Acols)

    if kmers_file is not None:
        kmers, labels = read_kmers(kmers_file, chromo, pos)
        X = np.hstack((X, kmers))
        cols.extend(labels)

    cols = [x.encode() for x in cols]
    return {'X': X, 'y': y, 'pos': pos, 'columns': cols}


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
            'data_file',
            help='HDF path to data')
        p.add_argument(
            '--annos_file',
            help='HDF path to annotation file')
        p.add_argument(
            '--annos_excl',
            help='Annotations to be excluded',
            nargs='+'
        )
        p.add_argument(
            '--kmers_file',
            help='HDF path to kmers file')
        p.add_argument(
            '-o', '--out_file',
            help='Output file')
        p.add_argument(
            '--chromos',
            help='Chromosomes',
            nargs='+')
        p.add_argument(
            '--knn',
            help='Number of KNNs',
            type=int,
            default=2)
        p.add_argument(
            '--knn_group',
            help='Name of KNN group',
            default='knn')
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

        log.info('Read')
        d = dict()
        chromos = opts.chromos
        if chromos is None:
            f = h5.File(opts.data_file)
            chromos = list(f['cpg'].keys())
            f.close()
        chromos_len = []
        for chromo in chromos:
            log.info('Chromosome %s' % (chromo))
            dc = read_chromo(opts.data_file, chromo,
                             annos_file=opts.annos_file,
                             max_samples=opts.max_samples,
                             knn=opts.knn,
                             knn_group=opts.knn_group,
                             annos_excl=opts.annos_excl,
                             kmers_file=opts.kmers_file)
            chromos_len.append(len(dc['y']))
            for k, v in dc.items():
                if k not in d.keys():
                    d[k] = []
                d[k].append(v)

        log.info('Write')
        for k, v in d.items():
            if k == 'columns':
                d[k] = v[0]
            elif v[0].ndim == 1:
                d[k] = np.hstack(v)
            else:
                d[k] = np.vstack(v)
        d['chromos'] = [x.encode() for x in chromos]
        d['chromos_len'] = np.array(chromos_len, dtype='int32')

        f = h5.File(opts.out_file, 'a')
        for k, v in d.items():
            if k in f.keys():
                del f[k]
            f.create_dataset(k, data=v)
        f.close()

        log.info('Done!')

        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
