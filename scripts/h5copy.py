#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
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
            description='Copies source HDF5 files to target file')
        p.add_argument(
            'src_files',
            help='Source files',
            nargs='+')
        p.add_argument(
            'dst_file',
            help='Destination file')
        p.add_argument(
            '-a', '--append',
            help='Append to file',
            action='store_true')
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

        mode = 'w'
        if opts.append:
            mode = 'a'
        dst_file = h5.File(opts.dst_file, mode)
        for src_path in opts.src_files:
            log.info(src_path)
            src_file = h5.File(src_path, 'r')
            for k in src_file.keys():
                if k in dst_file:
                    del dst_file[k]
                src_file.copy(k, dst_file)
            src_file.close()
        dst_file.close()

        log.info('Done!')
        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
