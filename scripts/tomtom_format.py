#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import pandas as pd


__dir = pt.dirname(pt.realpath(__file__))
# sys.path.insert(0, pt.join(__dir, '../module'))


def parse_meme(meme_file):
    motifs = dict()
    f = open(meme_file, 'r')
    for line in f:
        line = line.strip()
        if line.startswith('MOTIF'):
            line = line.split()
            if len(line) < 3:
                continue
            mid = line[1]
            motifs[mid] = [line[2], '']
        elif line.startswith('URL'):
            line = line.split()
            motifs[mid][1] = line[1]
    f.close()
    return motifs


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
            description='Formats tomtom output file')
        p.add_argument(
            'tomtom_file',
            help='Tomtom input file')
        p.add_argument(
            '-m', '--meme_db',
            help='meme database files',
            nargs='+')
        p.add_argument(
            '-o', '--out_file',
            help='Output file')
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

        motifs = dict()
        for meme_file in opts.meme_db:
            log.info('Parse %s' % (meme_file))
            motifs.update(parse_meme(meme_file))

        log.info('Enrich %s' % (opts.tomtom_file))
        tom = pd.read_table(opts.tomtom_file)
        tom.rename(columns={'#Query ID': 'Query ID'}, inplace=True)
        tom['Target name'] = ''
        tom['URL'] = ''
        for i in tom.index:
            target = tom.loc[i, 'Target ID']
            if target in motifs:
                tom.loc[i, 'Target name'] = motifs[target][0]
                tom.loc[i, 'URL'] = motifs[target][1]

        t = tom.to_csv(opts.out_file, sep='\t', index=False)
        if t is not None:
            print(t, end='')

        log.info('Done!')
        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
