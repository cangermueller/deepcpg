"""Functions for reading and matching annotations."""

from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
from six.moves import range


def read_bed(filename, sort=False, usecols=[0, 1, 2], *args, **kwargs):
    """Read chromo,start,end from BED file without formatting chromo."""
    d = pd.read_table(filename, header=None, usecols=usecols, *args, **kwargs)
    d.columns = range(d.shape[1])
    d.rename(columns={0: 'chromo', 1: 'start', 2: 'end'}, inplace=True)
    if sort:
        d.sort(['chromo', 'start', 'end'], inplace=True)
    return d


def in_which(x, ys, ye):
    """Return index of positions in intervals.

    Returns for positions `x[i]` index `j`, s.t. `ys[j] <= x[i] <= ye[j]`, or
    -1 if `x[i]` is not overlapped by any interval.
    Intervals must be non-overlapping!

    Parameters
    ----------
    x : list
        list of positions.
    ys: list
        list with start of interval sorted in ascending order.
    ye: list
        list with end of interval.

    Returns
    -------
    :class:`numpy.ndarray`
        n:class:`numpy.ndarray` with indices of overlapping intervals or -1.
    """
    n = len(ys)
    m = len(x)
    rv = np.empty(m, dtype=np.int)
    rv.fill(-1)
    i = 0
    j = 0
    while i < n and j < m:
        while j < m and x[j] <= ye[i]:
            if x[j] >= ys[i]:
                rv[j] = i
            j += 1
        i += 1
    return rv


def is_in(pos, start, end):
    """Test if position is overlapped by at least one interval."""
    return in_which(pos, start, end) >= 0


def distance(pos, start, end):
    """Return shortest distance between a position and a list of intervals.

    Parameters
    ----------
    pos: list
        List of integer positions.
    start: list
        Start position of intervals.
    end: list
        End position of intervals.

    Returns
    :class:`numpy.ndarray`
        :class:`numpy.ndarray` of same length as `pos` with shortest distance
        between each `pos[i]` and any interval.
    """
    m = len(start)
    n = len(pos)
    i = 0
    j = 0
    end_prev = -10**7
    dist = np.zeros(n)
    while i < m and j < n:
        while j < n and pos[j] <= end[i]:
            if pos[j] < start[i]:
                dist[j] = min(pos[j] - end_prev, start[i] - pos[j])
            j += 1
        end_prev = end[i]
        i += 1
    dist[j:] = pos[j:] - end_prev
    assert np.all(dist >= 0)
    return dist


def join_overlapping(s, e):
    """Join overlapping intervals.

    Transforms a list of possible overlapping intervals into non-overlapping
    intervals.

    Parameters
    ----------
    s : list
        List with start of interval sorted in ascending order
    e : list
        List with end of interval.

    Returns
    -------
    tuple
        `tuple` (s, e) of non-overlapping intervals.
    """
    rs = []
    re = []
    n = len(s)
    if n == 0:
        return (rs, re)
    l = s[0]
    r = e[0]
    for i in range(1, n):
        if s[i] > r:
            rs.append(l)
            re.append(r)
            l = s[i]
            r = e[i]
        else:
            r = max(r, e[i])
    rs.append(l)
    re.append(r)
    return (rs, re)


def join_overlapping_frame(d):
    """Join overlapping intervals of Pandas DataFrame.

    Uses `join_overlapping` to join overlapping intervals of
    :class:`pandas.DataFrame` `d`.
    """
    d = d.sort_values(['chromo', 'start', 'end'])
    e = []
    for chromo in d.chromo.unique():
        dc = d.loc[d.chromo == chromo]
        start, end = join_overlapping(dc.start.values, dc.end.values)
        ec = pd.DataFrame(dict(chromo=chromo, start=start, end=end))
        e.append(ec)
    e = pd.concat(e)
    e = e.loc[:, ['chromo', 'start', 'end']]
    return e


def group_overlapping(s, e):
    """Assign group index of indices.

    Assigns group index to intervals. Overlapping intervals will be assigned
    to the same group.

    Parameters
    ----------
    s : list
        list with start of interval sorted in ascending order.
    e : list
        list with end of interval.

    Returns
    -------
    :class:`numpy.ndarray`
        :class:`numpy.ndarray` with group indices.
    """
    n = len(s)
    group = np.zeros(n, dtype='int32')
    if n == 0:
        return group
    idx = 0
    r = e[0]
    for i in range(1, n):
        if s[i] > r:
            idx += 1
            r = e[i]
        else:
            r = max(r, e[i])
        group[i] = idx
    return group


def extend_len(start, end, min_len, min_pos=1):
    """Extend intervals to minimum length.

    Extends intervals `start`-`end` with length smaller than `min_len` to length
    `min_len` by equally decreasing `start` and increasing `end`.

    Parameters
    ----------
    start: list
        List of start position of intervals.
    end: list
        List of end position of intervals.
    min_len: int
        Minimum interval length.
    min_pos: int
        Minimum position.

    Returns
    -------
    tuple
        `tuple` with start end end position of extended intervals.
    """
    delta = np.maximum(0, min_len - (end - start + 1))
    ext = np.floor(0.5 * delta).astype(np.int)
    start_ext = np.maximum(min_pos, start - ext)
    end_ext = end + np.maximum(0, (min_len - (end - start_ext + 1)))
    assert np.all(min_len <= (end_ext - start_ext + 1))
    return (start_ext, end_ext)


def extend_len_frame(d, min_len):
    """Extend length of intervals in Pandas DataFrame using `extend_len`."""
    start, end = extend_len(d.start.values, d.end.values, min_len)
    e = d.copy()
    e['start'] = start
    e['end'] = end
    return e
