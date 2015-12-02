#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import pandas as pd
import sqlite3 as sql
import hashlib


def to_sql(sql_path, data, table, meta):
    md5 = hashlib.md5()
    for v in sorted(meta.values()):
        md5.update(v.encode())
    id_ = md5.hexdigest()

    data = data.copy()
    for k, v in meta.items():
        data[k] = v
    data['id'] = id_
    con = sql.connect(sql_path)
    try:
        con.execute('DELETE FROM %s WHERE id = "%s"' % (table, id_))
    except sql.OperationalError:
        pass
    data.to_sql(table, con, if_exists='append', index=False)
    con.close()


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
            description='Compare model performance')
        p.add_argument(
            'model_dirs',
            help='Model directories',
            nargs='+')
        p.add_argument(
            '-o', '--out_file',
            help='SQL output file',
            default='perf.sql')
        p.add_argument(
            '--sql_meta',
            help='Meta columns in SQL table',
            nargs='+')
        p.add_argument(
            '--seed',
            help='Seed of rng',
            type=int,
            default=0)
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

        sql_meta = dict()
        if opts.sql_meta is not None:
            for meta in opts.sql_meta:
                k, v = meta.split('=')
                sql_meta[k] = v

        for model_dir in opts.model_dirs:
            model = pt.basename(model_dir)
            sql_meta['path'] = pt.realpath(model_dir)
            sql_meta['model'] = model
            t = pt.join(model_dir, 'perf_val.csv')
            if pt.isfile(t):
                d = pd.read_table(t)
                to_sql(opts.out_file, d, 'perf_val', sql_meta)
            t = pt.join(model_dir, 'lc.csv')
            if pt.isfile(t):
                d = pd.read_table(t)
                to_sql(opts.out_file, d, 'lc', sql_meta)

        con = sql.connect(opts.out_file)
        pf = pd.read_sql('SELECT model, target, auc, tpr, tnr, mcc, cor' +\
                        ' FROM perf_val', con)
        con.close()

        t = pf.groupby('model', as_index=False).mean()
        t.sort_values('auc', inplace=True, ascending=False)
        print(t.to_string(index=False))

        return 0

if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
