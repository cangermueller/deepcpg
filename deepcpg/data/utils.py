from collections import Counter, OrderedDict
import gzip

import h5py as h5
import numpy as np
import pandas as pd
import re

from ..utils import EPS, filter_regex

CPG_NAN = -1


def get_output_stats(data_files, output_names, nb_sample=None):
    names = {'outputs': output_names}
    stats = OrderedDict()
    # batch_size large to reliably estimate statistics from batches
    reader = h5_reader(data_files, names, loop=False, shuffle=False,
                       batch_size=100000,
                       nb_sample=nb_sample)
    nb_batch = 0
    for batch in reader:
        for name in output_names:
            output = batch['outputs/%s' % name]
            stat = stats.setdefault(name, Counter())
            stat['nb_tot'] += len(output)
            stat['frac_obs'] += np.sum(output != CPG_NAN)
            output = np.ma.masked_values(output, CPG_NAN)
            stat['mean'] += output.mean()
            stat['var'] += output.var()
        nb_batch += 1

    for stat in stats.values():
        for key in ['mean', 'var']:
            stat[key] /= (nb_batch + EPS)
        stat['frac_obs'] /= (stat['nb_tot'] + EPS)
    return stats


def get_nb_sample(data_files, nb_max=None, batch_size=None):
    nb_sample = 0
    for data_file in data_files:
        data_file = h5.File(data_file, 'r')
        nb_sample += len(data_file['pos'])
        data_file.close()
        if nb_max and nb_sample > nb_max:
            nb_sample = nb_max
            break
    if batch_size:
        nb_sample = (nb_sample // batch_size) * batch_size
    return nb_sample


def get_dna_wlen(data_file, max_len=None):
    data_file = h5.File(data_file, 'r')
    wlen = data_file['/inputs/dna'].shape[1]
    if max_len:
        wlen = min(max_len, wlen)
    return wlen


def get_cpg_wlen(data_file, max_len=None):
    data_file = h5.File(data_file, 'r')
    group = data_file['/inputs/cpg']
    wlen = group['%s/dist' % list(group.keys())[0]].shape[1]
    if max_len:
        wlen = min(max_len, wlen)
    return wlen


def read_cpg_table(filename, chromos=None, nrows=None, round=True, sort=True):
    d = pd.read_table(filename, header=None, usecols=[0, 1, 2], nrows=nrows,
                      dtype={0: np.str, 1: np.int32, 2: np.float32},
                      comment='#')
    d.columns = ['chromo', 'pos', 'value']
    if chromos is not None:
        if not isinstance(chromos, list):
            chromos = [str(chromos)]
        d = d.loc[d.chromo.isin(chromos)]
    if sort:
        d.sort_values(['chromo', 'pos'], inplace=True)
    if round:
        d['value'] = np.round(d.value)
        if not np.all((d.value == 0) | (d.value == 1)):
            raise 'Invalid methylation states'
    return d


def add_to_dict(src, dst):
    for key, value in src.items():
        if isinstance(value, dict):
            if key not in dst:
                dst[key] = dict()
            add_to_dict(value, dst[key])
        else:
            if key not in dst:
                dst[key] = []
            dst[key].append(value)


def stack_dict(data):
    sdata = dict()
    for key, value in data.items():
        if isinstance(value, dict):
            sdata[key] = stack_dict(value)
        else:
            fun = np.vstack if value[0].ndim > 1 else np.hstack
            sdata[key] = fun(value)
    return sdata


def _h5_ls(item, recursive=False, groups=False, level=0):
    keys = []
    if isinstance(item, h5.Group):
        if groups and level > 0:
            keys.append(item.name)
        if level == 0 or recursive:
            for key in list(item.keys()):
                keys.extend(_h5_ls(item[key], recursive, groups, level + 1))
    elif not groups:
        keys.append(item.name)
    return keys


def h5_ls(filename, group='/', recursive=False, groups=False,
          regex=None, nb_keys=None):
    if not group.startswith('/'):
        group = '/%s' % group
    h5_file = h5.File(filename, 'r')
    keys = _h5_ls(h5_file[group], recursive, groups)
    for i, key in enumerate(keys):
        keys[i] = re.sub('^%s/' % group, '', key)
    h5_file.close()
    if regex:
        keys = filter_regex(keys, regex)
    if nb_keys:
        keys = keys[:nb_keys]
    return keys


def get_output_names(data_file, *args, **kwargs):
    return h5_ls(data_file, 'outputs',
                 recursive=True,
                 groups=False,
                 *args, **kwargs)


def get_replicate_names(data_file, *args, **kwargs):
    return h5_ls(data_file, 'inputs/cpg',
                 recursive=False,
                 groups=True,
                 *args, **kwargs)


def h5_write_data(data, filename):
    is_root = isinstance(filename, str)
    group = h5.File(filename, 'w') if is_root else filename
    for key, value in data.items():
        if isinstance(value, dict):
            key_group = group.create_group(key)
            h5_write_data(value, key_group)
        else:
            group[key] = value
    if is_root:
        group.close()


def h5_hnames_to_names(hnames):
    names = []
    for key, value in hnames.items():
        if isinstance(value, dict):
            for name in h5_hnames_to_names(value):
                names.append('%s/%s' % (key, name))
        elif isinstance(value, list):
            for name in value:
                names.append('%s/%s' % (key, name))
        elif isinstance(value, str):
            names.append('%s/%s' % (key, value))
        else:
            names.append(key)
    return names


def h5_read(data_files, names, batch_size=1024, *args, **kwargs):
    data = dict()
    reader = h5_reader(data_files, names, batch_size=batch_size, loop=False,
                       *args, **kwargs)
    for data_batch in reader:
        for key, value in data_batch.items():
            values = data.setdefault(key, [])
            values.append(value)
    data = stack_dict(data)
    return data


def h5_read_from(reader, nb_sample=None):
    data = dict()
    nb_seen = 0
    for data_batch in reader:
        for key, value in data_batch.items():
            values = data.setdefault(key, [])
            values.append(value)
        nb_seen += len(list(data_batch.values())[0])
        if nb_sample and nb_seen >= nb_sample:
            break

    data = stack_dict(data)
    if nb_sample:
        for key, value in data.items():
            data[key] = value[:nb_sample]

    return data


def h5_reader(data_files, names, batch_size=128, nb_sample=None, shuffle=False,
              loop=False):
    if not isinstance(data_files, list):
        data_files = [data_files]
    # Copy, since it might be changed by shuffling
    data_files = list(data_files)
    if isinstance(names, dict):
        names = h5_hnames_to_names(names)

    if nb_sample:
        # Select the first k files s.t. the total sample size is at least
        # nb_sample. Only these files will be shuffled.
        _data_files = []
        nb_seen = 0
        for data_file in data_files:
            h5_file = h5.File(data_file, 'r')
            nb_seen += len(h5_file[names[0]])
            h5_file.close()
            _data_files.append(data_file)
            if nb_seen >= nb_sample:
                break
        data_files = _data_files
    else:
        nb_sample = np.inf

    file_idx = 0
    nb_seen = 0
    while True:
        if shuffle and file_idx == 0:
            np.random.shuffle(data_files)
        h5_file = h5.File(data_files[file_idx], 'r')
        nb_sample_file = len(h5_file[names[0]])
        nb_batch = int(np.ceil(nb_sample_file / batch_size))

        for batch in range(nb_batch):
            batch_start = batch * batch_size
            nb_read = min(nb_sample - nb_seen, batch_size)
            batch_end = min(nb_sample_file, batch_start + nb_read)
            _batch_size = batch_end - batch_start
            if _batch_size == 0:
                break
            nb_seen += _batch_size

            data = dict()
            for name in names:
                data[name] = h5_file[name][batch_start:batch_end]
            yield data

            if nb_seen >= nb_sample:
                break

        h5_file.close()
        file_idx += 1
        assert nb_seen <= nb_sample
        if nb_sample == nb_seen:
            assert file_idx == len(data_files)
        if file_idx == len(data_files):
            if loop:
                file_idx = 0
                nb_seen = 0
            else:
                break


class GzipFile(object):

    def __init__(self, filename, mode='r', *args, **kwargs):
        self.is_gzip = filename.endswith('.gz')
        if self.is_gzip:
            self.fh = gzip.open(filename, mode, *args, **kwargs)
        else:
            self.fh = open(filename, mode, *args, **kwargs)

    def read(self, *args, **kwargs):
        tmp = self.fh.read(*args, **kwargs)
        return tmp

    def write(self, data):
        if self.is_gzip and isinstance(data, str):
            data = data.encode()
        self.fh.write(data)

    def close(self):
        self.fh.close()
