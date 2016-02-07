#!/usr/bin/env python

import argparse
import sys
import logging
import os
import os.path as pt
import h5py as h5

import predict.dna as dna


def get_seq(path, chromo, start, end):
    f = h5.File(path)
    s = f[str(chromo)].value[start:end + 1]
    f.close()
    return s


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
            description='Print genomic sequence mm10')
        p.add_argument(
            'chromo',
            help='Chromosome')
        p.add_argument(
            'start',
            help='Start position or window center (offset=1)',
            type=int)
        p.add_argument(
            'end',
            help='End position or window delta (offset=1)',
            type=int)
        p.add_argument(
            '-d', '--delta',
            help='Show sequence centered on position',
            action='store_true')
        p.add_argument(
            '--seq_file',
            help='Sequence file',
            default='mm10')
        p.add_argument(
            '--int',
            help='As integer sequence',
            action='store_true')
        p.add_argument(
            '--onehot',
            help='As one-hot encoded',
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

        if opts.seq_file == 'mm10':
            seq_file = pt.join(os.getenv('Pdata'), 'mm10.h5')
        else:
            seq_file = opts.seq_file
        chromo = str(opts.chromo)
        if opts.delta:
            pos = opts.start - 1
            start = pos - opts.end
            end = pos + opts.end
        else:
            start = opts.start - 1
            end = opts.end - 1
        s = get_seq(seq_file, chromo, start, end)
        print('%s (%d - %d)' % (chromo, start + 1, end + 1))
        print(s)
        if opts.int:
            print(dna.char2int(s))
        if opts.onehot:
            print(dna.int2onehot(dna.char2int(s)))

        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
