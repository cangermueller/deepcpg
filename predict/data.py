#!/usr/bin/env python

import os
import os.path as pt
import pandas as pd
import re
import subprocess as sp
import numpy as np

__dir = pt.dirname(pt.realpath(__file__))
import hdf


def get_pos(path, dataset, chromo):
    d = read_cpg_list(path, dataset, chromo)
    d = pd.pivot_table(d, index='pos', columns='sample', values='value')
    p = sorted(d.index)
    return p


def read_cpg_list(path, dataset, chromo, samples=None, nrows=None):
    group = pt.join(dataset, 'cpg', str(chromo))
    if samples is None:
        samples = hdf.ls(path, group)
    d = []
    for sample in samples:
        ds = pd.read_hdf(path, pt.join(group, sample), stop=nrows)
        ds['sample'] = sample
        d.append(ds)
    d = pd.concat(d)
    return d


def list_chromos(path, dataset):
    group = pt.join(dataset, 'pos')
    return hdf.ls(path, group)


def read_cpg(path, chromo=None, nrows=None):
    if chromo is not None:
        cmd = "grep '^\s*%s' %s" % (chromo, path)
        f = sp.Popen(cmd, shell=True, cwd=os.getcwd(), stdout=sp.PIPE).stdout
    else:
        f = path
    d = pd.read_table(f, header=None, usecols=[0, 1, 2], nrows=nrows,
                      dtype={0: np.str, 1: np.int32, 2: np.float32})
    d.columns = ['chromo', 'pos', 'value']
    d['chromo'] = [chromo_to_int(x) for x in d.chromo]
    d['value'] = np.round(d.value)
    assert np.all((d.value == 0) | (d.value == 1)), 'Invalid methylation states'
    d = pd.DataFrame(d, dtype=np.int32)
    return d


def read_annos(filename, sort=True):
    d = pd.read_table(filename, header=None)
    d = d.iloc[:, :3]
    d.columns = ['chromo', 'start', 'end']
    d['chromo'] = format_chromos(d['chromo'])
    if sort:
        d.sort(['chromo', 'start'], inplace=True)
    return d


def chromo_to_int(chromo):
    if type(chromo) is int:
        return chromo
    chromo = chromo.lower()
    if chromo == 'x':
        return 100
    elif chromo == 'y':
        return 101
    elif chromo in ['mt', 'm']:
        return 102
    else:
        return int(chromo)


def format_chromo(chromo, to_int=True):
    if type(chromo) is int:
        return chromo
    chromo = chromo.lower()
    chromo = re.sub('^chr', '', chromo)
    if to_int:
        chromo = chromo_to_int(chromo)
    return chromo


def format_chromos(chromos, *args, **kwargs):
    return [format_chromo(x, *args, **kwargs) for x in chromos]
