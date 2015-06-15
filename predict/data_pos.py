import argparse
import sys
import logging
import os.path as pt
import pandas as pd
import warnings

from predict import hdf
from predict import data


def add_pos(path, dataset):
    chromos = hdf.ls(path, pt.join(dataset, 'cpg'))
    for chromo in chromos:
        p = data.get_pos(path, dataset, chromo)
        p = pd.Series(p)
        group = pt.join(dataset, 'pos', chromo)
        p.to_hdf(path, group, format='t', data_columns=True)
