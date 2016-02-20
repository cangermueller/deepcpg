#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import numpy as np
from sklearn.manifold import TSNE

def sort_cpos(d):
    for k, v in d.items():
        d[k] = np.asarray(v)
    dc = {k: v.copy() for k, v in d.items()}
    chromos = np.unique(d['chromo'])
    chromos.sort()
    i = 0
    for chromo in chromos:
        h = d['chromo'] == chromo
        j = i + h.sum()
        for k, v in d.items():
            dc[k][i:j] = v[h]
        h = np.argsort(dc['pos'][i:j])
        for k, v in dc.items():
            dc[k][i:j] = dc[k][i:j][h]
        i = j
    return dc


def test_sort_cpos():
    d = dict(
        chromo=np.arange(10)[::-1],
        pos=np.arange(100, 110)[::-1],
        x=np.arange(200, 210)[::-1]
        )
    ds = sort_cpos(d)
    for k, v in d.items():
        npt.assert_array_equal(ds[k], np.array(v[::-1]))

    d = dict(
        chromo=[5, 5, 2, 1, 2, 5, 1],
        pos=   [9, 2, 7, 5, 4, 1, 0],
        x=     [1, 2, 3, 4, 5, 6, 7]
        )
    e = dict(
        chromo=[1, 1, 2, 2, 5, 5, 5],
        pos   =[0, 5, 4, 7, 1, 2, 9],
        x     =[7, 4, 5, 3, 6, 2, 1]
        )
    ds = sort_cpos(d)
    for k, v in e.items():
        npt.assert_array_equal(ds[k], np.array(v))


def _read_stats(path, chromo, stats, pos=None):
    f = h5.File(path, 'r')
    g = f[str(chromo)]
    p = g['pos'].value
    d = np.vstack([g[x].value for x in stats]).T
    f.close()
    if pos is not None:
        t = np.in1d(p, pos)
        d = d[t]
        p = p[t]
        assert np.all(p == pos)
    return p, d


def read_stats(path, chromos=None, pos=None, stats=None):
    f = h5.File(path, 'r')
    if chromos is None:
        chromos = list(f.keys())
    elif not isinstance(chromos, list):
        chromos = [chromos]
    chromos = [str(x) for x in chromos]
    _stats = [x for x in f[chromos[0]] if x != 'pos']
    if stats is None:
        stats = _stats
    else:
        stats = ut.filter_regex(_stats, stats)
    ds = []
    ps = []
    for i, chromo in enumerate(chromos):
        cpos = None
        if pos is not None:
            cpos = pos[i]
        p, d = _read_stats(path, chromo, stats, cpos)
        ps.append(p)
        ds.append(d)
    ds = np.vstack(ds)
    return (chromos, ps, ds, stats)

def cpos_to_list(chromos, pos):
    _chromos = np.unique(np.asarray(chromos))
    _pos = []
    for c in _chromos:
        _pos.append(pos[chromos == c])
    return (_chromos, _pos)

def cpos_to_vec(chromos, pos):
    _chromos = []
    for i, p in enumerate(pos):
        _chromos.append([chromos[i]] * len(p))
    _chromos = np.hstack(_chromos)
    _pos = np.hstack(pos)
    return (_chromos, _pos)

class App(object):

    def run(self, args):
        name = pt.basename(args[0])
        parser = self.create_parser(name)
        opts = parser.parse_args(args[1:])
        self.opts = opts
        return self.main(name, opts)

    def create_parser(self, name):
        p = argparse.ArgumentParser(
            prog=name,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Visualizes TSNE embeddings')
        p.add_argument(
            'act_file',
            help='File with layer activations')
        p.add_argument(
            '--group',
            help='HDF group in act_file')
        p.add_argument(
            '-o', '--out_dir',
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

        if opts.seed is not None:
            np.random.seed(opts.seed)

        log.info('Done!')

        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
