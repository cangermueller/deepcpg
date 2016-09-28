from collections import OrderedDict


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
