#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import h5py as h5
import numpy as np

import predict.utils as ut


def read_prediction(path, target=None, chromos=None, what=['pos', 'z']):
    f = h5.File(path)
    if target is not None:
        g = f[target]
    else:
        g = f
    if chromos is None:
        chromos = list(g.keys())
    d = {x: [] for x in what}
    for chromo in chromos:
        for k in d.keys():
            d[k].append(g[chromo][k].value)
    f.close()
    for k in what:
        if k != 'pos':
            d[k] = np.hstack(d[k])
    d['chromos'] = chromos
    return d


def read_annos(annos_file, chromo, name, pos=None):
    f = h5.File(annos_file)
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


def read_stats(stats_file, chromo, stats, pos=None):
    f = h5.File(stats_file)
    g = f[chromo]
    d = {k: g[k].value for k in [stats, 'pos']}
    f.close()
    if pos is not None:
        t = np.in1d(d['pos'], pos)
        for k in d.keys():
            d[k] = d[k][t]
        assert np.all(d['pos'] == pos)
    return d['pos'], d[stats]


def write_sliced(data, path, group, slice_=None, nb_max=None):
    if slice_ is None:
        size = len(list(data.values())[0])
        slice_ = slice(0, size)
    data = {k: v[slice_] for k, v in data.items()}
    size = len(list(data.values())[0])

    if nb_max is not None and size > nb_max:
        idx = np.arange(size)
        np.random.shuffle(idx)
        idx = idx[:nb_max]
        for k, v in data.items():
            data[k] = v[idx]

    out_file = h5.File(path, 'a')
    for k, v in data.items():
        h = pt.join(group, k)
        # Checking if h exists leads to huge files
        out_file.create_dataset(h, data=v, compression='gzip')
    out_file.close()


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
            description='Reformat prediction')
        p.add_argument(
            'pred_file',
            help='Prediction file')
        p.add_argument(
            '-o', '--out_file',
            help='Output file')
        p.add_argument(
            '--targets',
            help='Filter targets',
            nargs='+')
        p.add_argument(
            '--annos_file',
            help='HDF file with annotations')
        p.add_argument(
            '--annos',
            help='Regex of annotations to be considered',
            default=[r'^loc_.+', r'^licr_.+', r'.*H3.*'],
            nargs='+')
        p.add_argument(
            '--stats_file',
            help='HDF file with statistics')
        p.add_argument(
            '--stats',
            help='Statistics to be considered',
            default=['cov', 'var', 'entropy',
                     'win_cov', 'win_var', 'win_entropy', 'win_dist',
                     'gc_content', 'cg_obs_exp'],
            nargs='+')
        p.add_argument(
            '--stats_bins',
            help='Number of bins of quantization',
            type=int,
            default=5)
        p.add_argument(
            '--nb_sample',
            help='Sample that many observations',
            type=int)
        p.add_argument(
            '--chromos',
            help='Only consider these chromosomes',
            nargs='+')
        p.add_argument(
            '--verbose',
            help='More detailed log messages',
            action='store_true')
        p.add_argument(
            '--log_file',
            help='Write log messages to file')
        return p

    def _process(self, target):
        opts = self.opts
        log = self.log

        data = read_prediction(opts.pred_file, target=target,
                               chromos=opts.chromos,
                               what=['pos', 'y', 'z'])
        cpos = data['pos']
        chromos = data['chromos']
        del data['pos']
        del data['chromos']

        log.info('Global')
        write_sliced(data, opts.out_file, '%s/global' % (target),
                     nb_max=opts.nb_sample)

        if opts.annos_file is not None:
            f = h5.File(opts.annos_file)
            annos = list(f[chromos[0]].keys())
            f.close()
            annos = ut.filter_regex(annos, opts.annos)
            for anno in annos:
                log.info('Anno: %s' % (anno))
                idx = []
                for chromo, pos in zip(chromos, cpos):
                    idx.append(read_annos(opts.annos_file, chromo,
                                          anno, pos)[1])
                idx = np.hstack(idx)
                if idx.sum() > 10:
                    h = '%s/annos/%s' % (target, anno)
                    write_sliced(data, opts.out_file, h, idx,
                                 nb_max=opts.nb_sample)

        if opts.stats_file is not None:
            if opts.stats is None:
                f = h5.File(opts.stats_file)
                stats = f[chromos[0]].keys()
                f.close()
                stats = list(filter(lambda x: x != 'pos', stats))
            for stat in stats:
                log.info('Stat: %s' % (stat))
                idx = []
                for chromo, pos in zip(chromos, cpos):
                    idx.append(read_stats(opts.stats_file,
                                          chromo, stat, pos)[1])
                idx = np.hstack(idx)

                nb_bin = opts.stats_bins
                while nb_bin > 0:
                    try:
                        bins = ut.qcut(ut.add_noise(idx), nb_bin)
                        break
                    except ValueError:
                        nb_bin -= 1
                if nb_bin == 0:
                    raise ValueError('Insufficient observations ' +
                                     'for binning statistic!')

                for bin_ in bins.categories:
                    idx = bins == bin_
                    h = '%s/stats/%s/%s' % (target, stat, bin_)
                    write_sliced(data, opts.out_file, h, idx,
                                 nb_max=opts.nb_sample)

    def main(self, name, opts):
        logging.basicConfig(filename=opts.log_file,
                            format='%(levelname)s (%(asctime)s): %(message)s')
        log = logging.getLogger(name)
        if opts.verbose:
            log.setLevel(logging.DEBUG)
        else:
            log.setLevel(logging.INFO)
        log.debug(opts)

        self.log = log
        self.opts = opts

        f = h5.File(opts.pred_file, 'r')
        targets = list(f.keys())
        f.close()
        if opts.targets:
            targets = sorted(ut.filter_regex(targets, opts.targets))
        for target in targets:
            log.info('')
            log.info('Process %s' % (target))
            self._process(target)
        log.info('Done!')
        return 0

if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
