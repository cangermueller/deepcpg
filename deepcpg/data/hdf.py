"""Functions for accessing HDF5 files."""

from __future__ import division
from __future__ import print_function

import re

import h5py as h5
import numpy as np
import six
from six.moves import range

from ..utils import filter_regex, to_list


def _ls(item, recursive=False, groups=False, level=0):
    keys = []
    if isinstance(item, h5.Group):
        if groups and level > 0:
            keys.append(item.name)
        if level == 0 or recursive:
            for key in list(item.keys()):
                keys.extend(_ls(item[key], recursive, groups, level + 1))
    elif not groups:
        keys.append(item.name)
    return keys


def ls(filename, group='/', recursive=False, groups=False,
       regex=None, nb_key=None, must_exist=True):
    """List name of records HDF5 file.

    Parameters
    ----------
    filename:
        Path of HDF5 file.
    group:
        HDF5 group to be explored.
    recursive: bool
        If `True`, list records recursively.
    groups: bool
        If `True`, only list group names but not name of datasets.
    regex: str
        Regex to filter listed records.
    nb_key: int
        Maximum number of records to be listed.
    must_exist: bool
        If `False`, return `None` if file or group does not exist.

    Returns
    -------
    list
        `list` with name of records in `filename`.
    """
    if not group.startswith('/'):
        group = '/%s' % group
    h5_file = h5.File(filename, 'r')
    if not must_exist and group not in h5_file:
        return None
    keys = _ls(h5_file[group], recursive, groups)
    for i, key in enumerate(keys):
        keys[i] = re.sub('^%s/' % group, '', key)
    h5_file.close()
    if regex:
        keys = filter_regex(keys, regex)
    if nb_key is not None:
        keys = keys[:nb_key]
    return keys


def write_data(data, filename):
    """Write data in dict `data` to HDF5 file."""
    is_root = isinstance(filename, str)
    group = h5.File(filename, 'w') if is_root else filename
    for key, value in six.iteritems(data):
        if isinstance(value, dict):
            key_group = group.create_group(key)
            write_data(value, key_group)
        else:
            group[key] = value
    if is_root:
        group.close()


def hnames_to_names(hnames):
    """Flattens `dict` `hnames` of hierarchical names.

    Converts hierarchical `dict`, e.g. hnames={'a': ['a1', 'a2'], 'b'}, to flat
    list of keys for accessing HDF5 file, e.g. ['a/a1', 'a/a2', 'b']
    """
    names = []
    for key, value in six.iteritems(hnames):
        if isinstance(value, dict):
            for name in hnames_to_names(value):
                names.append('%s/%s' % (key, name))
        elif isinstance(value, list):
            for name in value:
                names.append('%s/%s' % (key, name))
        elif isinstance(value, str):
            names.append('%s/%s' % (key, value))
        else:
            names.append(key)
    return names


def reader(data_files, names, batch_size=128, nb_sample=None, shuffle=False,
           loop=False):
    if isinstance(names, dict):
        names = hnames_to_names(names)
    else:
        names = to_list(names)
    # Copy, since list will be changed if shuffle=True
    data_files = list(to_list(data_files))

    # Check if names exist
    h5_file = h5.File(data_files[0], 'r')
    for name in names:
        if name not in h5_file:
            raise ValueError('%s does not exist!' % name)
    h5_file.close()

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
        data_file = dict()
        for name in names:
            data_file[name] = h5_file[name]
        nb_sample_file = len(list(data_file.values())[0])

        if shuffle:
            # Shuffle data within the entire file, which requires reading
            # the entire file into memory
            idx = np.arange(nb_sample_file)
            np.random.shuffle(idx)
            for name, value in six.iteritems(data_file):
                data_file[name] = value[:len(idx)][idx]

        nb_batch = int(np.ceil(nb_sample_file / batch_size))
        for batch in range(nb_batch):
            batch_start = batch * batch_size
            nb_read = min(nb_sample - nb_seen, batch_size)
            batch_end = min(nb_sample_file, batch_start + nb_read)
            _batch_size = batch_end - batch_start
            if _batch_size == 0:
                break

            data_batch = dict()
            for name in names:
                data_batch[name] = data_file[name][batch_start:batch_end]
            yield data_batch

            nb_seen += _batch_size
            if nb_seen >= nb_sample:
                break

        h5_file.close()
        file_idx += 1
        assert nb_seen <= nb_sample
        if nb_sample == nb_seen or file_idx == len(data_files):
            if loop:
                file_idx = 0
                nb_seen = 0
            else:
                break


def _to_dict(data):
    if isinstance(data, np.ndarray):
        data = [data]
    return dict(zip(range(len(data)), data))


def read_from(reader, nb_sample=None):
    from .utils import stack_dict

    data = dict()
    nb_seen = 0
    is_dict = True

    for data_batch in reader:
        if not isinstance(data_batch, dict):
            data_batch = _to_dict(data_batch)
            is_dict = False
        for key, value in six.iteritems(data_batch):
            values = data.setdefault(key, [])
            values.append(value)
        nb_seen += len(list(data_batch.values())[0])
        if nb_sample and nb_seen >= nb_sample:
            break

    data = stack_dict(data)
    if nb_sample:
        for key, value in six.iteritems(data):
            data[key] = value[:nb_sample]

    if not is_dict:
        data = [data[i] for i in range(len(data))]

    return data


def read(data_files, names, nb_sample=None, batch_size=1024, *args, **kwargs):
    data_reader = reader(data_files, names, batch_size=batch_size,
                         nb_sample=nb_sample, loop=False, *args, **kwargs)
    return read_from(data_reader, nb_sample)
