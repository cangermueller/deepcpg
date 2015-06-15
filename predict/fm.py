import argparse
import sys
import logging
import os.path as pt
import warnings
import pandas as pd

from predict import hdf
from predict import data_select as dsel


class Selector(object):
    """Selects data (X or Y matrix) from feature matrix."""

    def __init__(self, chromos=None):
        self.chromos = chromos

    def select_chromo(self, path, group, what, chromo):
        g = pt.join(group, what, chromo)
        d = pd.read_hdf(path, g)
        return d

    def select(self, path, group, what='X'):
        chromos = self.chromos
        if chromos is None:
            chromos = hdf.ls(path, pt.join(group, what))
        d = []
        for chromo in chromos:
            dc = self.select_chromo(path, group, what, chromo)
            d.append(dc)
        d = pd.concat(d, keys=chromos, names=['chromo', 'pos'])
        return d
