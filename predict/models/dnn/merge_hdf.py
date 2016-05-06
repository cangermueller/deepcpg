#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import h5py as h5
import numpy as np


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
            description='Merge split files')
        p.add_argument(
            'in_files',
            help='Input files',
            nargs='+')
        p.add_argument(
            '-o', '--out_file',
            help='Output file',
            default='merged.h5')
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

        dat = dict()
        for in_path in opts.in_files:
            log.info(in_path)
            in_file = h5.File(in_path, 'r')
            for k in list(in_file.keys()):
                dat.setdefault(k, [])
                dat[k].append(in_file[k].value)
        for k, v in dat.items():
            _ = np.vstack if v[0].ndim > 1 else np.hstack
            dat[k] = _(v)

        log.info('Write')
        assert np.all(dat['pos'] == np.sort(dat['pos']))
        out_file = h5.File(opts.out_file, 'w')
        for k, v in dat.items():
            out_file[k] = v
        out_file.close()
        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
