#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import sqlite3 as sql
import pandas as pd


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
            description='Copy models between databases')
        p.add_argument(
            'src_file',
            help='Source SQL file')
        p.add_argument(
            'dst_file',
            help='Destination SQL file')
        p.add_argument(
            '--models',
            help='Models to be copied',
            nargs='+')
        p.add_argument(
            '--tables',
            help='Database tables',
            nargs='+',
            default=['global', 'annos', 'stats'])
        p.add_argument(
            '--eval',
            help='Evaluation',
            default='e01')
        p.add_argument(
            '--trial',
            help='Trial',
            type=int,
            default=1)
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

        src_db = sql.connect(opts.src_file)
        dst_db = sql.connect(opts.dst_file)
        models = opts.models
        if models is None:
            models = list(pd.read_sql('SELECT distinct(model) from global',
                                      src_db).model)
        sel = 'model="{model}" AND eval="{ev}" AND trial="{trial}"'
        for table in opts.tables:
            try:
                cols = pd.read_sql('SELECT * FROM %s LIMIT 1' % (table), src_db)
            except pd.io.sql.DatabaseError:
                cols = []
            for model in models:
                sel = 'model="{model}" AND eval="{ev}" AND trial="{trial}"'
                sel = sel.format(model=model, ev=opts.eval, trial=opts.trial)
                cmd = 'SELECT * FROM %s WHERE %s' % (table, sel)
                d = pd.read_sql(cmd, src_db)
                if d.shape[0] > 0:
                    log.info('Copy %s.%s' % (table, model))
                    d = d.loc[:, list(set(cols) & set(d.columns))]
                    cmd = 'DELETE FROM %s WHERE %s' % (table, sel)
                    try:
                        dst_db.execute(cmd)
                    except sql.OperationalError:
                        pass
                    d.to_sql(table, dst_db, if_exists='append', index=False)
        src_db.close()
        dst_db.close()
        log.info('Done!')
        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
