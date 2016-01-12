#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import h5py as h5

import predict.models.dnn.model as mod


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
            description='Exports CNN filters to HDF5 file')
        p.add_argument(
            'model',
            help='Model files',
            nargs='+')
        p.add_argument(
            '-o', '--out_file',
            help='Output file',
            default='filter.h5')
        p.add_argument(
            '--out_group',
            help='Output group')
        p.add_argument(
            '--nodes',
            help='Convolutional nodes',
            nargs='+',
            default=['s_c1', 'c_c1']
        )
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

        m = mod.model_from_list(opts.model, compile=False)
        f = h5.File(opts.out_file, 'a')
        if opts.out_group is not None:
            if opts.out_group in f:
                del f[opts.out_group]
            fg = f.create_group(opts.out_group)
        else:
            fg = f
        for node in opts.nodes:
            if node in m.nodes:
                log.info('Export %s' % (node))
                fg[node] = m.nodes[node].get_weights()[0]
        f.close()

        log.info('Done!')
        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
