from collections import Counter, OrderedDict
import gzip
import threading

import h5py as h5
import numpy as np
import pandas as pd

from ..utils import EPS
from . import hdf

CPG_NAN = -1


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


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


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


def get_output_stats(data_files, output_names, nb_sample=None):
    names = {'outputs': output_names}
    stats = OrderedDict()
    # batch_size large to reliably estimate statistics from batches
    reader = hdf.reader(data_files, names, loop=False, shuffle=False,
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
            stat['mean'] += float(output.mean())
            stat['var'] += float(output.var())
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


def get_output_names(data_file, *args, **kwargs):
    return hdf.ls(data_file, 'outputs',
                  recursive=True,
                  groups=False,
                  *args, **kwargs)


def get_dna_wlen(data_file, max_len=None):
    data_file = h5.File(data_file, 'r')
    wlen = data_file['/inputs/dna'].shape[1]
    if max_len:
        wlen = min(max_len, wlen)
    return wlen


def get_replicate_names(data_file, *args, **kwargs):
    return hdf.ls(data_file, 'inputs/cpg',
                  recursive=False,
                  groups=True,
                  *args, **kwargs)


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
