import gzip

import h5py as h5
import numpy as np
import pandas as pd

from ..utils import filter_regex

CPG_NAN = -1


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


def h5_ls(data_file, group='/', keys_filter=None, nb_keys=None):
    data_file = h5.File(data_file, 'r')
    keys = list(data_file[group].keys())
    data_file.close()
    if keys_filter is not None:
        keys = filter_regex(keys, keys_filter)
    if nb_keys:
        keys = keys[:nb_keys]
    return keys


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


def h5_read(data_files, names, *args, **kwargs):
    data = dict()
    reader = h5_reader(data_files, names, loop=False, *args, **kwargs)
    for data_batch in reader:
        for key, value in data_batch.items():
            values = data.setdefault(key, [])
            values.append(value)
    data = stack_dict(data)
    return data


def h5_reader(data_files, names, batch_size=128, nb_sample=None, shuffle=False,
              loop=False):
    if not isinstance(data_files, list):
        data_files = [data_files]
    if isinstance(names, dict):
        names = h5_hnames_to_names(names)
    file_idx = 0
    nb_seen = 0
    if nb_sample is None:
        nb_sample = np.inf

    while True:
        if shuffle and file_idx == 0:
            np.random.shuffle(data_files)
        data_file = h5.File(data_files[file_idx], 'r')
        nb_sample_file = len(data_file[names[0]])
        nb_batch = int(np.ceil(nb_sample_file / batch_size))

        for batch in range(nb_batch):
            batch_start = batch * batch_size
            nb_read = min(nb_sample - nb_seen, batch_size)
            batch_end = min(nb_sample_file, batch_start + nb_read)
            nb_seen += batch_end - batch_start

            data = dict()
            for name in names:
                data[name] = data_file[name][batch_start:batch_end]
            yield data

            if nb_seen >= nb_sample:
                data_files = data_files[:file_idx + 1]
                break

        data_file.close()
        file_idx += 1
        if file_idx >= len(data_files):
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
