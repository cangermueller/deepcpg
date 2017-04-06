#!/usr/bin/env python

"""Export imputed methylation profiles.

Exports imputed methylation profiles from `dcpg_eval.py` output file to
different data formats. Outputs for each CpG site and cell either the
experimentally observed or predicted methylation state depending on whether or
not the methylation state was observed in the input file or not, respectively.
Creates for each methylation profile one file in the output directory.

Examples
--------
Export profiles of all cells as HDF5 files to `./eval`:

.. code:: bash

    dcpg_eval_export.py
        ./eval/data.h5
        --out_dir ./eval

Export the profile of cell Ca01 for chromosomes 4 and 5 to a bedGraph file:

.. code:: bash

    dcpg_eval_export.py
        ./eval/data.h5
        --output cpg/Ca01
        --chromo 4 5
        --format bedGraph
        --out_dir ./eval
"""

from __future__ import print_function
from __future__ import division

import os
import sys

import argparse
import h5py as h5
import logging
import numpy as np
import pandas as pd
import six

from deepcpg import data as dat
from deepcpg.utils import make_dir


def write_to_bedGraph(data, filename, compression=None):
    data = pd.DataFrame({'chromo': data['chromo'],
                         'start': data['pos'],
                         'end': data['pos'] + 1,
                         'value': data['value']},
                        columns=['chromo', 'start', 'end', 'value'])
    data['chromo'] = 'chr' + data['chromo'].str.decode('utf')
    data.to_csv(filename, sep='\t', index=False, header=None,
                float_format='%.5f',
                compression=compression)


def write_to_hdf(data, filename):
    out_file = h5.File(filename, 'w')
    for name in ['chromo', 'pos', 'value']:
        out_file.create_dataset(name, data=data[name], compression='gzip')
    out_file.close()


class App(object):

    def run(self, args):
        name = os.path.basename(args[0])
        parser = self.create_parser(name)
        opts = parser.parse_args(args[1:])
        return self.main(name, opts)

    def create_parser(self, name):
        p = argparse.ArgumentParser(
            prog=name,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Exports imputed methylation profiles')
        p.add_argument(
            'data_file',
            help='Output data file from `dcpg_eval.py`')
        p.add_argument(
            '-o', '--out_dir',
            help='Output directory',
            default='.')
        p.add_argument(
            '-f', '--out_format',
            help='Output file format',
            choices=['bedGraph', 'hdf'],
            default='hdf')
        p.add_argument(
            '--chromos',
            help='Chromosomes to be exported',
            nargs='+')
        p.add_argument(
            '--output_names',
            help='Regex to select outputs',
            nargs='+')
        p.add_argument(
            '--nb_sample',
            help='Number of samples',
            type=int)
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

        data_file = h5.File(opts.data_file, 'r')

        nb_sample = len(data_file['pos'])
        if opts.nb_sample:
            nb_sample = min(nb_sample, opts.nb_sample)

        data = dict()
        for name in ['chromo', 'pos']:
            data[name] = data_file[name][:nb_sample]

        idx = None
        if opts.chromos:
            idx = np.in1d(data['chromo'],
                          [chromo.encode() for chromo in opts.chromos])
            for key, value in six.iteritems(data):
                data[key] = value[idx]

        output_names = dat.get_output_names(opts.data_file,
                                            regex=opts.output_names)

        make_dir(opts.out_dir)

        for output_name in output_names:
            log.info(output_name)
            data['output'] = data_file['outputs'][output_name][:nb_sample]
            data['pred'] = data_file['preds'][output_name][:nb_sample]
            if idx is not None:
                for name in ['output', 'pred']:
                    data[name] = data[name][idx]

            # Use `output` label if known, otherwise prediction
            data['value'] = data['pred']
            tmp = data['output'] != dat.CPG_NAN
            data['value'][tmp] = data['output'][tmp]

            name = output_name.split(dat.OUTPUT_SEP)
            if name[0] == 'cpg':
                name = name[-1]
            else:
                name = '_'.join(name)
            out_file = os.path.join(opts.out_dir, name)

            if opts.out_format == 'bedGraph':
                write_to_bedGraph(data, out_file + '.bedGraph.gz',
                                  compression='gzip')
            elif opts.out_format == 'hdf':
                write_to_hdf(data, out_file + '.h5')
            else:
                tmp = 'Invalid output format "%s"!' % opts.out_format
                raise ValueError()

        log.info('Done!')

        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
