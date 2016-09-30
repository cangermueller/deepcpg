from collections import OrderedDict
import re

import numpy as np


def format_row(values, widths=None, sep=' | '):
    if widths:
        tmp = []
        for value, width in zip(values, widths):
            f = '%{0}s'.format(width)
            tmp.append(f % value)
        values = tmp
    return sep.join(values)


def format_table(table, colwidth=None, precision=2, header=True, sep=' | '):
    if not colwidth:
        colwidth = 0
    col_names = list(table.keys())
    col_widths = []
    tot_width = 0
    nb_row = None
    ftable = OrderedDict()
    for name in col_names:
        width = len(name)
        values = []
        for value in table[name]:
            if value is None:
                value = ''
            elif isinstance(value, float):
                value = '{0:.{1}f}'.format(value, precision)
            else:
                value = str(value)
            width = max(width, len(value))
            values.append(value)
        ftable[name] = values
        col_widths.append(width)
        if not nb_row:
            nb_row = len(values)
        else:
            nb_row = min(nb_row, len(values))
        tot_width += width
    tot_width += len(sep) * (len(col_widths) - 1)
    rows = []
    if header:
        rows.append(format_row(col_names, col_widths, sep=sep))
        rows.append('-' * tot_width)
    for row in range(nb_row):
        values = [values[row] for values in ftable.values()]
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
