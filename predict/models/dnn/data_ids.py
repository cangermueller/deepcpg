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
            description='Show target ids')
        p.add_argument(
            'data_file',
            help='Data file')
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

        in_file = h5.File(opts.data_file, 'r')
        g = in_file['targets']
        d = {x: g[x].value for x in ['id', 'name']}
        in_file.close()

        for k, v in d.items():
            d[k] = [x.decode() for x in v]
        for i in range(len(d['id'])):
            print('%s: %s' % (d['id'][i], d['name'][i]))

        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
