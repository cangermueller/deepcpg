import pandas as pd
import os.path as pt

__dir = pt.dirname(pt.realpath(__file__))

import hdf
import data_select
import eval_stats
import fm


class Selector(object):

    def __init__(self, chromos=None):
        self.chromos = chromos

    def select_Z(self, path, group):
        d = pd.read_hdf(path, group)
        if self.chromos:
            d = d.loc[self.chromos]
        return d

    def select_Y(self, path, group):
        sel = fm.Selector(self.chromos)
        d = sel.select(path, group, 'Y')
        return d

    def select_annos(self, path, group, annos=None):
        if annos is None:
            annos = True
        fsel = data_select.FeatureSelection()
        fsel.cpg = False
        fsel.knn = False
        fsel.knn_dist = False
        fsel.annos = True

        sel = data_select.Selector(fsel)
        sel.chromos = self.chromos

        d = sel.select(path, group)
        assert len(d.columns.levels[0]) == 1
        d.columns = d.columns.droplevel(0)
        return d

    def select_stats(self, path, group, stats=None):
        sel = eval_stats.Selector(self.chromos, stats)
        d = sel.select(path, group)
        return d
