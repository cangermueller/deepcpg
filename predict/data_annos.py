import argparse
import sys
import logging
import os.path as pt
import pandas as pd
import warnings

from predict import hdf
from predict import feature_extractor as fext
from predict import data
from predict import annos as A


class Processor(object):

    def __init__(self, path, dataset, distance=False):
        self.path = path
        self.dataset = dataset
        self.distance = distance

    def annotate(self, chromo, annos):
        pos = data.get_pos(self.path, self.dataset, chromo)
        chromo = int(chromo)
        annos = annos.loc[annos.chromo == chromo]
        start, end = A.join_overlapping(annos['start'].values,
                                            annos['end'].values)
        if self.distance:
            f = A.distance(pos, start, end)
        else:
            f = A.is_in(pos, start, end)
        f = pd.DataFrame(dict(pos=pos, value=f))
        return f

    def process_chromo(self, chromo, annos, anno_name):
        f = self.annotate(chromo, annos)
        group = 'annos'
        if self.distance:
            group += '_dist'
        out_group = pt.join(self.dataset, group, anno_name, chromo)
        f.to_hdf(self.path, out_group, format='t', data_columns=True)

    def process(self, annos, anno_name):
        chromos = data.list_chromos(self.path, self.dataset)
        for chromo in chromos:
            self.process_chromo(chromo, annos, anno_name)
