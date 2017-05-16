"""General-purpose functions."""

from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import os
import re
import six
from six.moves import range

import numpy as np

EPS = 10e-8


def make_dir(dirname):
    """Create directory `dirname` if non-existing.

    Parameters
    ----------
    dirname: str
        Path of directory to be created.

    Returns
    -------
    bool
        `True`, if directory did not exist and was created.
    """
    if os.path.exists(dirname):
        return False
    else:
        os.makedirs(dirname, exist_ok=True)
        return True


def slice_dict(data, idx):
    """Slice elements in dict `data` by `idx`.

    Slices array-like objects in `data` by index `idx`. `data` can be
    tree-like with sub-dicts, where the leafs must be sliceable by `idx`.

    Parameters
    ----------
    data: dict
        dict to be sliced.
    idx: slice
        Slice index.

    Returns
    -------
    dict
        dict with same elements as in `data` with sliced by `idx`.
    """
    if isinstance(data, dict):
        data_sliced = dict()
        for key, value in six.iteritems(data):
            data_sliced[key] = slice_dict(value, idx)
        return data_sliced
    else:
        return data[idx]


def fold_dict(data, nb_level=10**5):
    """Fold dict `data`.

    Turns dictionary keys, e.g. 'level1/level2/level3', into sub-dicts, e.g.
    data['level1']['level2']['level3'].

    Parameters
    ----------
    data: dict
        dict to be folded.
    nb_level: int
        Maximum recursion depth.

    Returns
    -------
    dict
        Folded dict.
    """
    if nb_level <= 0:
        return data

    groups = dict()
    levels = set()
    for key, value in data.items():
        idx = key.find('/')
        if idx > 0:
            level = key[:idx]
            group_dict = groups.setdefault(level, dict())
            group_dict[key[(idx + 1):]] = value
            levels.add(level)
        else:
            groups[key] = value
    for level in levels:
        groups[level] = fold_dict(groups[level], nb_level - 1)
    return groups


def linear_weights(length, start=0.1):
    """Create linear-triangle weights.

    Create array `x` of length `length` with linear weights, where the weight is
    highest (one) for the center x[length//2] and lowest (`start` ) at the ends
    x[0] and x[-1].

    Parameters
    ----------
    length: int
        Length of the weight array.
    start: float
        Minimum weights.

    Returns
    -------
    :class:`np.ndarray`
        Array of length `length` with weight.
    """
    weights = np.linspace(start, 1, np.ceil(length / 2))
    tmp = weights
    if length % 2:
        tmp = tmp[:-1]
    weights = np.hstack((weights, tmp[::-1]))
    return weights


def to_list(value):
    """Convert `value` to a list."""
    if not isinstance(value, list) and value is not None:
        value = [value]
    return value


def move_columns_front(frame, columns):
    """Move `columns` of Pandas DataFrame to the front."""
    if not isinstance(columns, list):
        columns = [columns]
    columns = [column for column in columns if column in frame.columns]
    return frame[columns + list(frame.columns[~frame.columns.isin(columns)])]


def get_from_module(identifier, module_params, ignore_case=True):
    """Return object from module.

    Return object with name `identifier` from module with items `module_params`.

    Parameters
    ----------
    identifier: str
        Name of object, e.g. a function, in module.
    module_params: dict
        `dict` of items in module, e.g. `globals()`
    ignore_case: bool
        If `True`, ignore case of `identifier`.

    Returns
    -------
    object
        Object with name `identifier` in module, e.g. a function or class.
    """
    if ignore_case:
        _module_params = dict()
        for key, value in six.iteritems(module_params):
            _module_params[key.lower()] = value
        _identifier = identifier.lower()
    else:
        _module_params = module_params
        _identifier = identifier
    item = _module_params.get(_identifier)
    if not item:
        raise ValueError('Invalid identifier "%s"!' % identifier)
    return item


def format_table_row(values, widths=None, sep=' | '):
    """Format a row with `values` of a table."""
    if widths:
        _values = []
        for value, width in zip(values, widths):
            if value is None:
                value = ''
            _values.append('{0:>{1}s}'.format(value, width))
    return sep.join(_values)


def format_table(table, colwidth=None, precision=2, header=True, sep=' | '):
    """Format a table of values as string.

    Formats a table represented as a `dict` with keys as column headers and
    values as a lists of values in each column.

    Parameters
    ----------
    table: `dict` or `OrderedDict`
        `dict` or `OrderedDict` with keys as column headers and values as lists
        of values in each column.
    precision: int or list of ints
        Precision of floating point values in each column. If `int`, uses same
        precision for all columns, otherwise formats columns with different
        precisions.
    header: bool
        If `True`, print column names.
    sep: str
        Column separator.

    Returns
    -------
    str
        String of formatted table values.
    """

    col_names = list(table.keys())
    if not isinstance(precision, list):
        precision = [precision] * len(col_names)
    col_widths = []
    tot_width = 0
    nb_row = None
    ftable = OrderedDict()
    for col_idx, col_name in enumerate(col_names):
        width = max(len(col_name), precision[col_idx] + 2)
        values = []
        for value in table[col_name]:
            if value is None:
                value = ''
            elif isinstance(value, float):
                value = '{0:.{1}f}'.format(value, precision[col_idx])
            else:
                value = str(value)
            width = max(width, len(value))
            values.append(value)
        ftable[col_name] = values
        col_widths.append(width)
        if not nb_row:
            nb_row = len(values)
        else:
            nb_row = max(nb_row, len(values))
        tot_width += width
    tot_width += len(sep) * (len(col_widths) - 1)
    rows = []
    if header:
        rows.append(format_table_row(col_names, col_widths, sep=sep))
        rows.append('-' * tot_width)
    for row in range(nb_row):
        values = []
        for col_values in six.itervalues(ftable):
            if row < len(col_values):
                values.append(col_values[row])
            else:
                values.append(None)
        rows.append(format_table_row(values, col_widths, sep=sep))
    return '\n'.join(rows)


def filter_regex(values, regexs):
    """Filters list of `values` by list of `regexs`.

    Paramters
    ---------
    values: list
        list of `str` values.
    regexs: list
        list of `str` regexs.

    Returns
    -------
    list
        Sorted `list` of values in `values` that match any regex in `regexs`.
    """
    if not isinstance(values, list):
        values = [values]
    if not isinstance(regexs, list):
        regexs = [regexs]
    filtered = set()
    for value in values:
        for regex in regexs:
            if re.search(regex, value):
                filtered.add(value)
    return sorted(list(filtered))


class ProgressBar(object):
    """Vertical progress bar.

    Unlike the progressbar2 package, logs progress as multiple lines instead of
    single line, which enables printing to a file. Used, for example, in

    Parameters
    ----------
    nb_tot: int
        Maximum value
    logger: function
        Function that takes a `str` and prints it.
    interval: float
        Logging frequency as fraction of one. For example, 0.1 logs every tenth
        value.

    See also
    --------
    dcpg_eval.py and dcpg_filter_act.py.
    """

    def __init__(self, nb_tot, logger=print, interval=0.1):
        if nb_tot <= 0:
            raise ValueError('Total value must be greater than zero!')
        self.nb_tot = nb_tot
        self.logger = logger
        self.interval = interval
        self._value = 0
        self._nb_interval = 0

    def update(self, amount):
        tricker = self._value == 0
        amount = min(amount, self.nb_tot - self._value)
        self._value += amount
        self._nb_interval += amount
        tricker |= self._nb_interval >= int(self.nb_tot * self.interval)
        tricker |= self._value >= self.nb_tot
        if tricker:
            nb_digit = int(np.floor(np.log10(self.nb_tot))) + 1
            msg = '{value:{nb_digit}d}/{nb_tot:d} ({per:3.1f}%)'
            msg = msg.format(value=self._value, nb_digit=nb_digit,
                             nb_tot=self.nb_tot,
                             per=self._value / self.nb_tot * 100)
            self.logger(msg)
            self._nb_interval = 0

    def close(self):
        if self._value < self.nb_tot:
            self.update(self.nb_tot)
