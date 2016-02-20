#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import numpy as np
import h5py as h5
import random
from sklearn.manifold import TSNE

import predict.io as io


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
            'in_file',
            help='File with layer activations')
        p.add_argument(
            '--in_group',
            help='Input HDF5 group')
        p.add_argument(
            '-o', '--out_file',
            help='Output HDF5 file')
        p.add_argument(
            '--annos_file',
            help='HDF file with annotations')
        p.add_argument(
            '--annos',
            help='Regex of annotations to be considered',
            default=[r'^.+$'],
            nargs='+')
        p.add_argument(
            '--stats_file',
            help='HDF file with statistics')
        p.add_argument(
            '--stats',
            help='Regex of statistics to be considered',
            default=['cov', 'var', 'entropy',
                     'win_cov', 'win_var', 'win_entropy', 'win_dist',
                     'gc_content', 'cg_obs_exp'],
            nargs='+')
        p.add_argument(
            '--nb_sample',
            help='Maximum number of samples',
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
            random.seed(opts.seed)

        log.info('Read activations')
        in_file = h5.File(opts.in_file, 'r')
        group = in_file
        if opts.in_group is not None:
            group = group[opts.in_group]
        nb_sample = group['pos'].shape[0]
        if opts.nb_sample is not None:
            nb_sample = min(opts.nb_sample, nb_sample)

        act = {x: group[x][:nb_sample] for x in ['chromo', 'pos', 'act']}
        in_file.close()
        act = io.sort_cpos(act)

        log.info('Compute TSNE embedding')
        tsne = TSNE()
        act['act'] = tsne.fit_transform(act['act'])

        log.info('Write TSNE')
        out_file = h5.File(opts.out_file, 'w')
        out_file['act'] = act['act']
        out_file['pos'] = act['pos']
        out_file['chromos'] = act['chromo']

        chromos, pos = io.cpos_to_list(act['chromo'], act['pos'])
        chromos = [x.decode() for x in chromos]

        if opts.stats_file is not None:
            log.info('Write stats')
            c, p, d, n = io.read_stats(opts.stats_file, chromos, pos,
                                       opts.stats)
            g = out_file.create_group('stats')
            for i, nn in enumerate(n):
                g[nn] = d[:, i]

        if opts.annos_file is not None:
            log.info('Write annos')
            c, p, d, n = io.read_annos(opts.annos_file, chromos, pos,
                                       opts.annos)
            g = out_file.create_group('annos')
            for i, nn in enumerate(n):
                g[nn] = d[:, i]

        log.info('Done!')
        out_file.close()

        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
