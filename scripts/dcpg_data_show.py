#!/usr/bin/env python

"""Show the content of DeepCpG data files.

Shows the content of ``dcpg_data.py`` output files for a selected region, for
example the methylation state of the target CpG site, neighboring CpG sites, or
the DNA sequence.

Examples
--------
Show the output methylation state of CpG sites on on chromosome 19 between
position 3028955 and 3079682:

.. code:: bash

    dcpg_data_show.py
        ./data/*.h5
        --chromo 1
        --start 3028955
        --end 3079682
        --outputs

Show output methylation states and the state as well as the distance of
10 neighboring CpG sites of cell BS27_1_SER:

.. code:: bash

    dcpg_data_show.py
        ./data/*.h5
        --chromo 1
        --start 3028955
        --end 3079682
        --outputs cpg/BS27_1_SER
        --cpg BS27_1_SER
        --cpg_wlen 10
        --cpg_dist

Show output methylation states and DNA sequence windows of length 11 and
store the results in HDF5 file ``selected.h5``:

.. code:: bash

    dcpg_data_show.py
        ./data/*.h5
        --chromo 1
        --start 3028955
        --end 3079682
        --outputs
        --dna_wlen 11
        --out_hdf selected.h5
"""

from __future__ import print_function
from __future__ import division

from collections import OrderedDict
import os
import sys

import argparse
import h5py as h5
import logging
import numpy.random
import pandas as pd

from deepcpg.data import hdf


def delta_columns(delta, zero=True):
    columns = ['%s' % column for column in list(range(-delta, 0))]
    if zero:
        columns.append('0')
    columns += ['+%s' % column for column in list(range(1, delta + 1))]
    return columns


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
            description='Shows selected data from DeepCpG data files')
        p.add_argument(
            'data_files',
            nargs='+',
            help='Data files')
        p.add_argument(
            '-o', '--out_hdf',
            help='Write selected data to HDF5 file')
        p.add_argument(
            '--outputs',
            nargs='*',
            help='Name of outputs that are shown')
        p.add_argument(
            '--dna_wlen',
            type=int,
            help='Length of DNA sequence window that is shown')
        p.add_argument(
            '--cpg',
            nargs='*',
            help='Name of replicates that are shown')
        p.add_argument(
            '--cpg_wlen',
            type=int,
            default=10,
            help='CpG window length')
        p.add_argument(
            '--cpg_dist',
            action='store_true',
            help='Also show distance to neighboring CpG sites')
        p.add_argument(
            '--chromo',
            help='Chromosome')
        p.add_argument(
            '--start',
            type=int,
            help='Start position')
        p.add_argument(
            '--end',
            type=int,
            help='End position')
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

        if opts.dna_wlen and opts.dna_wlen % 2 == 0:
            raise ValueError('DNA window length must be odd!')

        if opts.cpg_wlen and opts.cpg_wlen % 2 == 1:
            raise ValueError('CpG window length must be even!')

        data = []
        for filename in opts.data_files:
            data_file = h5.File(filename, 'r')
            data_chunk = OrderedDict()
            loc = pd.DataFrame({'chromo': data_file['chromo'].value,
                                'pos': data_file['pos'].value},
                               columns=['chromo', 'pos'])
            data_chunk['loc'] = loc

            if opts.outputs is not None:
                group = data_file['outputs']
                output_names = opts.outputs
                if not len(output_names):
                    output_names = hdf.ls(filename, 'outputs', recursive=True)
                outputs = []
                for output_name in output_names:
                    output = pd.Series(group[output_name].value,
                                       name=output_name)
                    outputs.append(output)
                outputs = pd.concat(outputs, axis=1)
                data_chunk['outputs'] = outputs

            if opts.dna_wlen:
                group = data_file['/inputs/dna']
                wlen = group.shape[1]
                delta = opts.dna_wlen // 2
                ctr = wlen // 2
                idx = slice(ctr - delta, ctr + delta + 1)
                dna = group[:, idx]
                dna = pd.DataFrame(dna, columns=delta_columns(delta))
                data_chunk['dna'] = dna

            if opts.cpg is not None:
                kinds = ['state']
                if opts.cpg_dist:
                    kinds.append('dist')

                group = data_file['/inputs/cpg']
                names = opts.cpg
                if not len(names):
                    names = list(group.keys())
                for name in names:
                    for kind in kinds:
                        path = '%s/%s' % (name, kind)
                        cpg = group[path].value
                        if opts.cpg_wlen:
                            ctr = cpg.shape[1] // 2
                            delta = opts.cpg_wlen // 2
                            cpg = cpg[:, (ctr - delta):(ctr + delta)]
                        columns = delta_columns(delta, zero=False)
                        cpg = pd.DataFrame(cpg, columns=columns)
                        data_chunk[path] = cpg

            data_chunk = pd.concat(data_chunk.values(),
                                   axis=1,
                                   keys=data_chunk.keys())
            if opts.chromo:
                idx = data_chunk['loc']['chromo'] == opts.chromo.encode()
                data_chunk = data_chunk.loc[idx]
            if opts.start:
                idx = data_chunk['loc']['pos'] >= opts.start
                data_chunk = data_chunk.loc[idx]
            if opts.end:
                idx = data_chunk['loc']['pos'] <= opts.end
                data_chunk = data_chunk.loc[idx]
            data.append(data_chunk)

        data = pd.concat(data)
        data[('loc', 'chromo')] = [x.decode() for x in data['loc']['chromo']]
        if opts.out_hdf:
            data.to_hdf(opts.out_hdf, '/data')
        else:
            print(data.to_string())

        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
