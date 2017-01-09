from collections import OrderedDict
import os
import re

import numpy as np

EPS = 10e-8


def make_dir(dirname):
    if os.path.exists(dirname):
        return False
    else:
        os.makedirs(dirname, exist_ok=True)
        return True


def slice_dict(data, idx):
    if isinstance(data, dict):
        data_sliced = dict()
        for key, value in data.items():
            data_sliced[key] = slice_dict(value, idx)
        return data_sliced
    else:
        return data[idx]


def linear_weights(length, start=0.1):
    weights = np.linspace(start, 1, np.ceil(length / 2))
    tmp = weights
    if length % 2:
        tmp = tmp[:-1]
    weights = np.hstack((weights, tmp[::-1]))
    return weights


def to_list(value):
    if not isinstance(value, list) and value is not None:
        value = [value]
    return value


def move_columns_front(frame, columns):
    if not isinstance(columns, list):
        columns = [columns]
    columns = [column for column in columns if column in frame.columns]
    return frame[columns + list(frame.columns[~frame.columns.isin(columns)])]


def get_from_module(identifier, module_params, ignore_case=True):
    if ignore_case:
        _module_params = dict()
        for key, value in module_params.items():
            _module_params[key.lower()] = value
        _identifier = identifier.lower()
    else:
        _module_params = module_params
        _identifier = identifier
    item = _module_params.get(_identifier)
    if not item:
        raise ValueError('Invalid identifier "%s"!' % identifier)
    return item


def format_row(values, widths=None, sep=' | '):
    if widths:
        _values = []
        for value, width in zip(values, widths):
            if value is None:
                value = ''
            _values.append('{0:>{1}s}'.format(value, width))
    return sep.join(_values)


def format_table(table, colwidth=None, precision=2, header=True, sep=' | '):
    if not colwidth:
        colwidth = 0
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
        rows.append(format_row(col_names, col_widths, sep=sep))
        rows.append('-' * tot_width)
    for row in range(nb_row):
        values = []
        for col_values in ftable.values():
            if row < len(col_values):
                values.append(col_values[row])
            else:
                values.append(None)
        rows.append(format_row(values, col_widths, sep=sep))
    return '\n'.join(rows)


def filter_regex(x, regexs):
    if not isinstance(x, list):
        x = [x]
    if not isinstance(regexs, list):
        regexs = [regexs]
    xf = set()
    for xi in x:
        for regex in regexs:
            if re.search(regex, xi):
                xf.add(xi)
    return sorted(list(xf))


class ProgressBar(object):

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
