import pandas as pd
import numpy as np


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


def join_index(index, sep='_'):
    return [sep.join(x) for x in index.values]


def to_rhdf(d, filename, group):
    if d.columns.nlevels > 1:
        d = d.copy()
        d.columns = join_index(d.columns)
    d = d.reset_index()
    d.to_hdf(filename, group, format='t', data_columns=True)
