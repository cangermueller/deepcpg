#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import numpy as np
import h5py as h5


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
            'stats_files',
            help='Statistic files',
            nargs='+')
        p.add_argument(
            '--stats_names',
            help='Names for be used for statistics files',
            nargs='+')
        p.add_argument(
            '-o', '--out_file',
            help='Output HDF path')
        p.add_argument(
            '--min_cov',
            help='Require at least one observation per site',
            action='store_true')
        p.add_argument(
            '--chromos',
            help='Only apply to these chromosome',
            nargs='+')
        p.add_argument(
            '--nb_sample',
            help='Only consider that many samples',
            type=int)
        p.add_argument(
            '--verbose', help='More detailed log messages', action='store_true')
        p.add_argument(
            '--log_file', help='Write log messages to file')
        return p

    def join(self, chromo, opts, log):
        log.info('Join positions')
        pos = None
        for stats_file in opts.stats_files:
            f = h5.File(stats_file, 'r')
            sp = f[chromo]['pos'].value
            if opts.min_cov:
                sc = f[chromo]['cov'].value
                sp = sp[sc > 0]
            f.close()
            if pos is None:
                pos = sp
            else:
                pos = pos[np.in1d(pos, sp)]
        log.info('%d positions' % (len(pos)))

        out_file = h5.File(opts.out_file, 'a')
        if chromo in out_file:
            del out_file[chromo]
        out_group = out_file.create_group(chromo)
        out_group['pos'] = pos
        for s, stats_file in enumerate(opts.stats_files):
            log.info('Join %s' % (stats_file))
            in_file = h5.File(stats_file, 'r')
            in_group = in_file[chromo]
            if opts.stats_names is None:
                stat_name = pt.splitext(pt.basename(stats_file))[0]
            else:
                stat_name = opts.stats_names[s]
            idx = np.in1d(in_group['pos'], pos)
            for stat in in_group.keys():
                if stat in ['pos']:
                    continue
                h = '%s_%s' % (stat_name, stat)
                out_group[h] = in_group[stat].value[idx]
            in_file.close()
        out_file.close()

    def main(self, name, opts):
        logging.basicConfig(filename=opts.log_file,
                            format='%(levelname)s (%(asctime)s): %(message)s')
        log = logging.getLogger(name)
        if opts.verbose:
            log.setLevel(logging.DEBUG)
        else:
            log.setLevel(logging.INFO)
        log.debug(opts)

        chromos = []
        for stat_file in opts.stats_files:
            f = h5.File(stat_file, 'r')
            chromos.append(list(f.keys()))
            f.close()
        c = set(chromos[0])
        for i in range(1, len(chromos)):
            c = c & set(chromos[i])
        if opts.chromos is not None:
            c = c & set(opts.chromos)
        chromos = sorted(list(c))
        for chromo in chromos:
            log.info('Join chromo %s' % (chromo))
            self.join(chromo, opts, log)
        log.info('Done!')
        return 0

if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
