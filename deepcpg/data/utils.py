"""General purpose IO functions."""

from __future__ import division
from __future__ import print_function

import gzip
import threading
import re

import h5py as h5
import numpy as np
import pandas as pd
import six
from six.moves import range

from . import hdf

# Constant for missing labels.
CPG_NAN = -1
# Constant for separating output names, e.g. 'cpg/cell'.
OUTPUT_SEP = '/'


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)

    def next(self):
        return self.__next__()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe."""
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


def add_to_dict(src, dst):
    """Add `dict `src` to `dict` `dst`

    Adds values in `dict` `src` to `dict` `dst` with same keys but values are
    lists of added values. lists of values in `dst` can be stacked with
    :func:`stack_dict`.  Used for example in `dpcg_eval.py` to stack dicts from
    different batches.

    Example
    -------
    src = dict()
    src['a'] = 1
    src['b'] = {'b1': 10}
    dst = dict()
    add_to_dict(src, dst)
    add_to_dict(src, dst)
    -> dst['a'] = [1, 1]
    -> dst['b'] = {'b1': [10, 10]}
    """
    for key, value in six.iteritems(src):
        if isinstance(value, dict):
            if key not in dst:
                dst[key] = dict()
            add_to_dict(value, dst[key])
        else:
            if key not in dst:
                dst[key] = []
            dst[key].append(value)


def stack_dict(data):
    """Stacks lists of numpy arrays in `dict` `data`."""
    sdata = dict()
    for key, value in six.iteritems(data):
        if isinstance(value, dict):
            sdata[key] = stack_dict(value)
        else:
            fun = np.vstack if value[0].ndim > 1 else np.hstack
            sdata[key] = fun(value)
    return sdata


def get_nb_sample(data_files, nb_max=None, batch_size=None):
    """Count number of samples in all `data_files`.

    Parameters
    ----------
    data_files: list
        `list` with file name of DeepCpG data files.
    nb_max: int
        If defined, stop counting if that number is reached.
    batch_size: int
        If defined, return the largest multiple of `batch_size` that is smaller
        or equal than the actual number of samples.

    Returns
    -------
    int
        Number of samples in `data_files`.
    """
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
    """Return length of DNA sequence windows stored in `data_file`."""
    data_file = h5.File(data_file, 'r')
    wlen = data_file['/inputs/dna'].shape[1]
    if max_len:
        wlen = min(max_len, wlen)
    return wlen


def get_cpg_wlen(data_file, max_len=None):
    """Return number of CpG neighbors stored in `data_file`."""
    data_file = h5.File(data_file, 'r')
    group = data_file['/inputs/cpg']
    wlen = group['%s/dist' % list(group.keys())[0]].shape[1]
    if max_len:
        wlen = min(max_len, wlen)
    return wlen


def get_output_names(data_file, *args, **kwargs):
    """Return name of outputs stored in `data_file`."""
    return hdf.ls(data_file, 'outputs',
                  recursive=True,
                  groups=False,
                  *args, **kwargs)


def get_replicate_names(data_file, *args, **kwargs):
    """Return name of replicates stored in `data_file`."""
    return hdf.ls(data_file, 'inputs/cpg',
                  recursive=False,
                  groups=True,
                  must_exist=False,
                  *args, **kwargs)


def get_anno_names(data_file, *args, **kwargs):
    """Return name of annotations stored in `data_file`."""
    return hdf.ls(data_file, 'inputs/annos',
                  recursive=False,
                  *args, **kwargs)


def is_bedgraph(filename):
    """Test if `filename` is a bedGraph file.

    bedGraph files are assumed to start with 'track type=bedGraph'
    """
    if isinstance(filename, str):
        with open(filename) as f:
            line = f.readline()
    else:
        pos = filename.tell()
        line = filename.readline()
        if isinstance(line, bytes):
            line = line.decode()
        filename.seek(pos)
    return re.match(r'track\s+type=bedGraph', line) is not None


def format_chromo(chromo):
    """Format chromosome name.

    Makes name upper case, e.g. 'mt' -> 'MT' and removes 'chr',
    e.g. 'chr1' -> '1'.
    """
    return chromo.str.upper().str.replace('^CHR', '')


def sample_from_chromo(frame, nb_sample):
    """Randomly sample `nb_sample` samples from each chromosome.

    Samples `nb_sample` records from :class:`pandas.DataFrame` which must
    contain a column with name 'chromo'.
    """
    def sample_frame(frame):
        if len(frame) <= nb_sample:
            return frame
        idx = np.random.choice(len(frame), nb_sample, replace=False)
        return frame.iloc[idx]

    frame = frame.groupby('chromo', as_index=False).apply(sample_frame)
    frame.index = range(len(frame))
    return frame


def is_binary(values):
    """Check if values in array `values` are binary, i.e. zero or one."""
    return ~np.any((values > 0) & (values < 1))


def read_cpg_profile(filename, chromos=None, nb_sample=None, round=False,
                     sort=True, nb_sample_chromo=None):
    """Read CpG profile from TSV or bedGraph file.

    Reads CpG profile from either tab delimited file with columns
    `chromo`, `pos`, `value`. `value` or bedGraph file. `value` columns contains
    methylation states, which can be binary or continuous.

    Parameters
    ----------
    filenamne: str
        Path of file.
    chromos: list
        List of formatted chromosomes to be read, e.g. ['1', 'X'].
    nb_sample: int
        Maximum number of sample in total.
    round: bool
        If `True`, round methylation states in column 'value' to zero or one.
    sort: bool
        If `True`, sort by rows by chromosome and position.
    nb_sample_chromo: int
        Maximum number of sample per chromosome.

    Returns
    -------
    :class:`pandas.DataFrame`
         :class:`pandas.DataFrame` with columns `chromo`, `pos`, `value`.
    """

    if is_bedgraph(filename):
        usecols = [0, 1, 3]
        skiprows = 1
    else:
        usecols = [0, 1, 2]
        skiprows = 0
    dtype = {usecols[0]: np.str, usecols[1]: np.int32, usecols[2]: np.float32}
    nrows = None
    if chromos is None and nb_sample_chromo is None:
        nrows = nb_sample
    d = pd.read_table(filename, header=None, comment='#', nrows=nrows,
                      usecols=usecols, dtype=dtype, skiprows=skiprows)
    d.columns = ['chromo', 'pos', 'value']
    if np.any((d['value'] < 0) | (d['value'] > 1)):
        raise ValueError('Methylation values must be between 0 and 1!')
    d['chromo'] = format_chromo(d['chromo'])
    if chromos is not None:
        if not isinstance(chromos, list):
            chromos = [str(chromos)]
        d = d.loc[d.chromo.isin(chromos)]
        if len(d) == 0:
            raise ValueError('No data available for selected chromosomes!')
    if nb_sample_chromo is not None:
        d = sample_from_chromo(d, nb_sample_chromo)
    if nb_sample is not None:
        d = d.iloc[:nb_sample]
    if sort:
        d.sort_values(['chromo', 'pos'], inplace=True)
    if round:
        d['value'] = np.round(d.value)
    if is_binary(d['value']):
        d['value'] = d['value'].astype(np.int8)
    return d


class GzipFile(object):
    """Wrapper to read and write gzip-compressed files.

    If `filename` ends with `gz`, opens file with gzip package, otherwise
    builtin `open` function.

    Parameters
    ----------
    filename: str
        Path of file
    mode: str
        File access mode
    *args: list
        Unnamed arguments passed to open function.
    **kwargs: dict
        Named arguments passed to open function.
    """
    def __init__(self, filename, mode='r', *args, **kwargs):
        self.is_gzip = filename.endswith('.gz')
        if self.is_gzip:
            self.fh = gzip.open(filename, mode, *args, **kwargs)
        else:
            self.fh = open(filename, mode, *args, **kwargs)

    def __iter__(self):
        return self.fh.__iter__()

    def __next__(self):
        return self.fh.__next__()

    def read(self, *args, **kwargs):
        return self.fh.read(*args, **kwargs)

    def readline(self, *args, **kwargs):
        return self.fh.readline(*args, **kwargs)

    def readlines(self, *args, **kwargs):
        return self.fh.readlines(*args, **kwargs)

    def write(self, data):
        if self.is_gzip and isinstance(data, str):
            data = data.encode()
        self.fh.write(data)

    def writelines(self, *args, **kwargs):
        self.fh.writelines(*args, **kwargs)

    def tell(self, *args, **kwargs):
        return self.fh.tell(*args, **kwargs)

    def seek(self, *args, **kwargs):
        self.fh.seek(*args, **kwargs)

    def closed(self):
        return self.fh.closed()

    def close(self):
        self.fh.close()

    def __iter__(self):
        self.fh.__iter__()

    def iter(self):
        self.fh.iter()
