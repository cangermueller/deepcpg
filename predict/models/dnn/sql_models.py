#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import pandas as pd
import sqlite3 as sql
import hashlib

from predict.models.dnn.model import Params


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


def to_field(v):
    if isinstance(v, dict):
        return str(v)
    else:
        return v

def model_to_tables(params):
    d = vars(params)
    sub = ['cpg', 'seq', 'target']
    f = {'model': dict()}
    for k in sorted(d.keys()):
        v = d[k]
        if k in sub:
            f['model'][k] = str(v.__class__).lower().find(k) > -1
        else:
            f['model'][k] = to_field(v)
    for s in sub:
        if f['model'][s]:
            dd = vars(d[s])
            f[s] = {k: to_field(dd[k]) for k in sorted(dd.keys())}
        else:
            f[s] = None
    return f


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
            description='Add model config files to SQL database')
        p.add_argument(
            'model_files',
            help='Model files',
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

        for model_file in opts.model_files:
            log.info(model_file)
            params = Params.from_yaml(model_file)
            d = model_to_tables(params)
            sql_meta['model'] = pt.splitext(pt.basename(model_file))[0]
            sql_meta['path'] = model_file
            for k, v in d.items():
                if v is None:
                    continue
                name = k
                if name != 'model':
                    name = 'model_' + name
                v = pd.DataFrame(v, columns=sorted(v.keys()), index=[0])
                to_sql(opts.out_file, v, name, sql_meta)

        con = sql.connect(opts.out_file)
        d = pd.read_sql('SELECT * FROM model ORDER BY model', con)
        con.close()
        print(d.to_string(index=False))

        return 0

if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
