#!/usr/bin/env python

import argparse
import sys
import logging
import pdb
import os.path as pt
import h5py
import re
import sys
import pandas as pd

from predict import hdf

def find_cpgs(seq, cpg='CG'):
    rv = [m.start() for m in re.finditer(cpg, seq)]
    return rv


class Cpgs(object):

    def run(self, args):
        name = pt.basename(args[0])
        p = argparse.ArgumentParser(prog=name,
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                    description='Extracts CpG positions from sequence')
        p.add_argument('seq_file', help='HDF path where chromosome seqs can be found')
        p.add_argument('-o', '--out_file', help='Output file')
        p.add_argument('--start', help='Index of first sequence character', default=1, type=int)
        p.add_argument('--verbose', help='More detailed log messages', action='store_true')
        p.add_argument('--log_file', help='Write log messages to file')
        opts = p.parse_args(args[1:])
        self.opts = opts

        logging.basicConfig(filename=opts.log_file,
                            format='%(levelname)s (%(asctime)s): %(message)s')
        log = logging.getLogger(name)
        if opts.verbose:
            log.setLevel(logging.DEBUG)
        else:
            log.setLevel(logging.INFO)
        log.debug(opts)
        self.log = log

        hdf_file, hdf_path = hdf.split_path(opts.seq_file)
        f = h5py.File(hdf_file)
        chromos = list(f[hdf_path].keys())
        cpgs = []
        for chromo in chromos:
            log.info('Chromosome %s ...' % (chromo))
            seq = f[pt.join(hdf_path, chromo)].value
            t = find_cpgs(seq)
            t = pd.DataFrame({'chromo': chromo, 'pos': t})
            cpgs.append(t)
        cpgs = pd.concat(cpgs)
        cpgs.sort(['chromo', 'pos'], inplace=True)
        cpgs['pos'] += opts.start
        log.info('%d CpG sites detected.' % (cpgs.shape[0]))

        log.info('Writing output ...')
        t = cpgs.to_csv(opts.out_file, sep='\t', index=False)
        if t is not None:
            print(t, end='')

        log.info('Done!')
        return 0


if __name__ == '__main__':
    Cpgs().run(sys.argv)
