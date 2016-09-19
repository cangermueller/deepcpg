#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import numpy as np
import pandas as pd
import h5py as h5


def read_cpg(path, chromos=None, nrows=None):
    d = pd.read_table(path, header=None, usecols=[0, 1, 2], nrows=nrows,
                      dtype={0: np.str, 1: np.int32, 2: np.float32})
    d.columns = ['chromo', 'pos', 'value']
    if chromos is not None:
        if not isinstance(chromos, list):
            chromos = [str(chromos)]
        d = d.loc[d.chromo.isin(chromos)]
    d.set_index(['chromo', 'pos'], inplace=True)
    d['value'] = np.round(d.value)
    assert np.all((d.value == 0) | (d.value == 1)), 'Invalid methylation states'
    d = pd.Series(d.value, index=d.index, dtype='int8')
    return d


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
            description='Import CpG into data file')
        p.add_argument(
            'in_file',
            help='Input files with methylation rates in bed format \
            (chromo pos rate).')
        p.add_argument(
            '-o', '--out_file',
            help='Output HDF path.',
            default='data.h5')
        p.add_argument(
            '--out_group',
            help='Output group in HDF file',
            default='cpg')
        p.add_argument(
            '--chromos',
            help='Only use these chromosomes',
            nargs='+')
        p.add_argument(
            '--nrows',
            help='Only read that many rows from file',
            type=int)
        p.add_argument(
            '--verbose', help='More detailed log messages', action='store_true')
        p.add_argument(
            '--log_file', help='Write log messages to file')
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

        log.info('Read data')
        cpg = read_cpg(opts.in_file, opts.chromos, opts.nrows)
        cpg = cpg.sort_index()

        hdf_file = h5.File(opts.out_file, 'a')
        for chromo in cpg.index.levels[0]:
            log.info('Chromosome %s' % (chromo))
            cpos = cpg.loc[chromo].index.values.astype('int32')
            ccpg = cpg.loc[chromo].values.astype('int8')

            def write(name, data):
                g = pt.join(opts.out_group, chromo, name)
                if g in hdf_file:
                    del hdf_file[g]
                hdf_file.create_dataset(g, data=data)

            write('pos', cpos)
            write('cpg', ccpg)
        hdf_file.close()
        log.info('Done')
        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
