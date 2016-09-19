#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import numpy as np
import pandas as pd
import h5py as h5

import predict.io as io


def lin_pos(pos, wlen=5):
    lp = []
    lr = []
    dlen = wlen // 2
    for p in pos:
        _ = np.arange(p - dlen, p + dlen + 1)
        lp.append(_)
        lr.append(np.abs(_ - p))
    lp = np.hstack(lp)
    lr = np.hstack(lr)
    return lp, lr


def lin_array(pos, xs, agg_fun='nearest', *args, **kwargs):
    # 1st col: position
    # 2nd col: relative position from center
    _ = list(lin_pos(pos, wlen=xs[0].shape[1], *args, **kwargs))
    for x in xs:
        if x.ndim == 3:
            x = x.reshape(x.shape[0] * x.shape[1], x.shape[2]).T
        else:
            x = x.ravel()
        _.append(x)
    _ = pd.DataFrame(np.vstack(_).T)
    _.sort_values([0, 1], inplace=True)
    _ = _.groupby(0, as_index=False)
    if agg_fun == 'mean':
        _ = _.mean()
    elif agg_fun == 'max':
        _ = _.max()
    else:
        _ = _.first()
    return _.values


def test_lin_arrary():
    p = np.array([5, 6, 10, 15])
    v = np.ones((len(p), 3)) * p.reshape(-1, 1)
    pv = lin_array(p, v, wlen=3)
    ep = [[4, 5, 6, 7,  9, 10, 11, 14, 15, 16],
          [1, 0, 0, 1,  1,  0,  1,  1,  0,  1],
          [5, 5, 6, 6, 10, 10, 10, 15, 15, 15]]
    ep = np.array(ep).T
    assert np.all(pv == ep)


def lin_array_chromo(chromos, cpos, xs, *args, **kwargs):
    o = 0
    ds = []
    cs = []
    for c, p in zip(chromos, cpos):
        xc = [x[o:(o + len(p))] for x in xs]
        d = lin_array(p, xc, *args, **kwargs)
        ds.append(d)
        cs.append([c] * len(d))
        o += len(p)

    cs = np.hstack(cs)
    ds = np.vstack(ds)
    ps = ds[:, 0]
    rs = ds[:, 1]
    ds = ds[:, 2:]
    return (cs, ps, rs, ds)


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
            description='Linearizes windowed data')
        p.add_argument(
            'data_file',
            help='Data file')
        p.add_argument(
            '-o', '--out_file',
            help='Output file',
            default='lin.h5')
        p.add_argument(
            '-d', '--datasets',
            help='Datasets',
            nargs='+')
        p.add_argument(
            '--agg_fun',
            help='Aggregation function',
            choices=['nearest', 'mean', 'max'],
            default='nearest')
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

        log.info('Load data')
        in_file = h5.File(opts.data_file, 'r')
        data = dict()
        data['chromo'] = in_file['chromo'].value
        data['pos'] = in_file['pos'].value
        data['x'] = [in_file[_].value for _ in opts.datasets]
        in_file.close()

        log.info('Linearize')
        chromos, cpos = io.cpos_to_list(data['chromo'], data['pos'])
        lin = lin_array_chromo(chromos, cpos, data['x'], agg_fun=opts.agg_fun)

        log.info('Write')
        out_file = h5.File(opts.out_file, 'w')
        out_file['chromo'] = lin[0]
        out_file['pos'] = lin[1]
        out_file['rpos'] = lin[2]
        ndim = lin[3].shape[1] // len(opts.datasets)
        for i, d in enumerate(opts.datasets):
            out_file[d] = lin[3][:, i * ndim: (i + 1) * ndim].squeeze()
        out_file.close()

        log.info('Done!')
        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
