#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import h5py as h5


def update_file(path):
    f = h5.File(path)
    if 'labels' in f:
        f.move('labels', 'targets')
        g = f['targets']
        if 'files' in g:
            g.move('files', 'names')
        if 'targets' in g:
            g.move('targets', 'ids')
    if 'targets' in f:
        g = f['targets']
        if 'names' in g:
            g.move('names', 'name')
        if 'ids' in g:
            g.move('ids', 'id')
    f.close()


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
            description='Update data file')
        p.add_argument(
            'data_file',
            help='Data file to be updated')
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

        log.info('Update %s' % (opts.data_file))
        update_file(opts.data_file)

        log.info('Done!')
        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
