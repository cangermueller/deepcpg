import pandas as pd
import numpy as np
import re
import hashlib
import sqlite3 as sql


def ranges_to_list(x, start=0, stop=None):
    s = set()
    for xi in x:
        xi = str(xi)
        if xi.find('-') >= 0:
            t = xi.split('-')
            if len(t) != 2:
                raise ValueError('Invalid range!')
            if len(t[0]) == 0:
                t[0] = start
            if len(t[1]) == 0:
                t[1] = stop
            s |= set(range(int(t[0]), int(t[1]) + 1))
        else:
            s.add(int(xi))
    s = sorted(list(s))
    return s


def sql_id(meta):
    md5 = hashlib.md5()
    for v in sorted(meta.values()):
        md5.update(v.encode())
    id_ = md5.hexdigest()
    return id_


def sql_exits(sql_path, id_, table):
    con = sql.connect(sql_path)
    cmd = con.execute('SELECT id FROM %s WHERE id = "%s"' % (table, id_))
    count = len(cmd.fetchall())
    con.close()
    return count > 0


def to_sql(sql_path, data, table, meta):
    id_ = sql_id(meta)
    data = data.copy()
    for k, v in meta.items():
        data[k] = v
    data['id'] = id_
    con = sql.connect(sql_path, timeout=999999)
    try:
        con.execute('DELETE FROM %s WHERE id = "%s"' % (table, id_))
        cols = pd.read_sql('SELECT * FROM %s LIMIT 1' % (table), con)
    except sql.OperationalError:
        cols = []
    if len(cols):
        t = sorted(set(data.columns) - set(cols.columns))
        if len(t):
            print('Ignoring columns %s' % (' '.join(t)))
            data = data.loc[:, cols.columns]
    data.to_sql(table, con, if_exists='append', index=False)
    con.close()


def add_noise(x, eps=1e-6):
    min_ = np.min(x)
    max_ = np.max(x)
    xeps = x + np.random.uniform(-eps, eps, len(x))
    xeps = np.maximum(min_, xeps)
    xeps = np.minimum(max_, xeps)
    return xeps


def qcut(x, nb_bin, *args, **kwargs):
    p = np.arange(0, 101, 100 / nb_bin)
    q = list(np.percentile(x, p))
    y = pd.cut(x, bins=q, include_lowest=True, *args, **kwargs)
    assert len(y.categories) == nb_bin
    assert y.isnull().any() == False
    return y


def within_01(x, eps=1e-6):
    return np.maximum(eps, np.minimum(1 - eps, x))


def logodds(p, q):
    p = within_01(p)
    q = within_01(q)
    return np.log2(p) - np.log2(q)


def logodds_ratio(p, q):
    p = within_01(p)
    q = within_01(q)
    return np.log2(p / (1 - p)) - np.log2(q / (1 - q))


def filter_regex(x, regexs):
    xf = []
    for xi in x:
        for regex in regexs:
            if re.search(regex, xi):
                xf.append(xi)
    return xf


def group_apply(d, by, fun, level=False, set_index=False, *args, **kwargs):
    if level:
        g = d.groupby(level=by)
    else:
        g = d.groupby(by)
    r_all = []
    for k in g.groups.keys():
        dg = g.get_group(k)
        r = fun(dg, *args, **kwargs)
        if type(by) is list:
            for i in range(len(by)):
                r[by[i]] = k[i]
        else:
            r[by] = k
        if set_index:
            r = r.set_index(by, append=True)
            r = r.swaplevel(0, r.index.nlevels-1)
        r_all.append(r)
    r_all = pd.concat(r_all)
    return r_all


def rolling_apply(d, delta, fun, level=None):
    rv = None
    l = 0
    r = 0
    if level is None:
        pos = d.index
    else:
        pos = d.index.get_level_values(level)
    n = len(pos)
    for i in range(n):
        p = pos[i]
        while l < i and p - pos[l] > delta:
            l += 1
        while r < len(pos) - 1 and pos[r + 1] - p <= delta:
            r += 1
        di = d.iloc[l:(r + 1)]
        rvi = np.atleast_1d(fun(di))
        if rv is None:
            rv = np.empty((n, rvi.shape[0]))
        rv[i] = rvi
    rv = pd.DataFrame(rv, index=d.index, columns=d.columns)
    return rv
