#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import pandas as pd
import numpy as np
import warnings

__dir = pt.dirname(pt.realpath(__file__))
# sys.path.insert(0, pt.join(__dir, '../module'))

import hdf
import feature_extractor as fext
import data
import annos as A


class Processor(object):

    def __init__(self, path, dataset):
        self.path = path
        self.dataset = dataset

    def annotate(self, chromo, annos):
        pos = data.get_pos(self.path, self.dataset, chromo)
        chromo = int(chromo)
        annos = annos.loc[annos.chromo == chromo]
        start, end = A.join_overlapping(annos['start'].values,
                                            annos['end'].values)
        f = np.empty(len(pos))
        f.fill(np.nan) # score is nan if not in any interval
        m = A.in_which(pos, start, end)
        f[m >= 0] = annos.iloc[m[m >= 0]].score
        f = pd.DataFrame(dict(pos=pos, value=f))
        return f

    def process_chromo(self, chromo, annos, anno_name):
        f = self.annotate(chromo, annos)
        out_group = pt.join(self.dataset, 'scores', anno_name, chromo)
        f.to_hdf(self.path, out_group, format='t', data_columns=True)

    def process(self, annos, anno_name):
        annos = annos.sort(['chromo', 'start', 'end'])
        chromos = data.list_chromos(self.path, self.dataset)
        for chromo in chromos:
            self.process_chromo(chromo, annos, anno_name)


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
            description='Add score annotations')
        p.add_argument(
            'in_file',
            help='Input HDF path to dataset (test, train, val)')
        p.add_argument(
            '-a', '--anno_files',
            help='Annotation files in BED format with score column',
            nargs='+')
        p.add_argument(
            '--score_col',
            help='Index of scores column (starting at 1)',
            default=4,
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

        log.info('Add score annotations ...')
        in_path, in_group = hdf.split_path(opts.in_file)
        p = Processor(in_path, in_group)
        for anno_file in opts.anno_files:
            anno_name = pt.splitext(pt.basename(anno_file))[0]
            log.info('\t%s...', anno_name)
            annos = pd.read_table(anno_file, header=None,
                              usecols=[0, 1, 2, opts.score_col - 1])
            annos.columns = ['chromo', 'start', 'end', 'score']
            annos = data.format_bed(annos)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                p.process(annos, anno_name)
        log.info('Done!')
        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
