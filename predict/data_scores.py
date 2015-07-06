import argparse
import sys
import logging
import os.path as pt
import pandas as pd
import numpy as np
import warnings

from predict import hdf
from predict import feature_extractor as fext
from predict import data
from predict import annos as A


class Processor(object):

    def __init__(self, path, dataset):
        self.path = path
        self.dataset = dataset

    def annotate(self, chromo, annos):
        pos = data.read_pos(self.path, self.dataset, chromo)
        chromo = int(chromo)
        annos = annos.loc[annos.chromo == chromo]
        start, end = A.join_overlapping(annos['start'].values,
                                            annos['end'].values)
        f = np.empty(len(pos))
        f.fill(np.nan) # score is nan if not in any interval
        m = A.in_which(pos, start, end)
        f[m >= 0] = annos.iloc[m[m >= 0]].score
        f = pd.DataFrame(dict(pos=pos, value=f))
        f.set_index('pos', inplace=True)
        return f

    def process_chromo(self, chromo, annos, anno_name):
        f = self.annotate(chromo, annos)
        out_group = pt.join(self.dataset, 'scores', anno_name, chromo)
        f.to_hdf(self.path, out_group)

    def process(self, annos, anno_name):
        annos = annos.sort(['chromo', 'start', 'end'])
        chromos = data.list_chromos(self.path, self.dataset)
        for chromo in chromos:
            self.process_chromo(chromo, annos, anno_name)
