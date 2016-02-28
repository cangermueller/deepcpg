#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import sqlite3 as sql
import pandas as pd

__dir = pt.dirname(pt.realpath(__file__))
# sys.path.insert(0, pt.join(__dir, '../module'))


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
            description='Query evaluation database')
        p.add_argument(
            'sql_file',
            help='SQL file')
        p.add_argument(
            '-n', '--nb_models',
            help='Maximum number of models',
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
        log.debug(opts)

        con = sql.connect(opts.sql_file)
        d = pd.read_sql('SELECT model, auc, tpr, tnr FROM global', con)
        con.close()

        d = d.groupby('model', as_index=False).mean()
        d = d.sort_values('auc', ascending=False)
        if opts.nb_models:
            d = d.iloc[:opts.nb_models]
        d.model = d.model.str.replace('dnn_', '')
        print(d.to_csv(None, header=False, index=False, sep='\t'), end='')

        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
