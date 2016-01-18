#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import h5py as h5
import pandas as pd
import pickle


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
            description='Extracts features importances')
        p.add_argument(
            'target_dirs',
            help='Target directories',
            nargs='+')
        p.add_argument(
            '-o', '--out_file',
            help='Output file')
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

        d = []
        for target_dir in opts.target_dirs:
            target = pt.basename(target_dir)
            if len(target) == 0:
                target = pt.basename(pt.dirname(target_dir))
            log.info(target)
            f = h5.File(pt.join(target_dir, 'data/train.h5'), 'r')
            cols = [x.decode() for x in f['/columns']]
            f.close()
            with open(pt.join(target_dir, 'train', 'model.pkl'), 'rb') as f:
                m = pickle.load(f)
            dt = m.feature_importances_.reshape(1, -1)
            dt = pd.DataFrame(dt, columns=cols, index=[target])
            d.append(dt)
        d = pd.concat(d)
        d.index.name = 'target'
        d = d.reset_index().sort_values('target')
        s = d.to_csv(opts.out_file, sep='\t', index=False)
        if s is not None:
            print(s, end='')

        log.info('Done!')

        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
