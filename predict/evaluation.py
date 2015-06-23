import pandas as pd

from predict import data_select
from predict import eval_stats
from predict import fm


class Loader(object):

    def __init__(self, group='test', chromos=None):
        self.chromos = chromos
        self.group = group

    def Y(self, data_path):
        sel = fm.Selector(self.chromos)
        d = sel.select(data_path, self.group, 'Y')
        return d

    def y(self, data_path):
        # [chromo, pos, sample, y]
        y = self.Y(data_path)
        y = pd.melt(y.reset_index(), id_vars=['chromo', 'pos'], var_name='sample', value_name='y').dropna()
        return y

    def Z(self, z_path):
        d = pd.read_hdf(z_path, self.group)
        if self.chromos:
            d = d.loc[self.chromos]
        return d

    def z(self, z_path):
        # [chromo, pos, sample, z]
        z = self.Z(z_path)
        z = pd.melt(z.reset_index(), id_vars=['chromo', 'pos'], var_name='sample', value_name='z')
        return z

    def A(self, data_path, annos=None):
        if annos is None:
            annos = True
        fsel = data_select.FeatureSelection()
        fsel.cpg = False
        fsel.knn = False
        fsel.knn_dist = False
        fsel.annos = annos

        sel = data_select.Selector(fsel)
        sel.chromos = self.chromos

        d = sel.select(data_path, self.group)
        assert len(d.columns.levels[0]) == 1
        d.columns = d.columns.droplevel(0)
        return d

    def a(self, data_path, annos=None):
        # [chromo, pos, anno]
        a = self.A(data_path, annos)
        a = pd.melt(a.reset_index(), id_vars=['chromo', 'pos'], var_name='anno', value_name='is_in')
        a = a.assign(is_in=a.is_in == 1).query('is_in == True')
        a = a.loc[:, a.columns != 'is_in']
        return a

    def S(self, es_path, stats=None):
        sel = eval_stats.Selector(self.chromos, stats)
        d = sel.select(es_path, self.group)
        return d

    def s(self, es_path, stats=None):
        # [chromo, pos, stat, value]
        s = self.S(es_path, stats)
        s = pd.melt(s.reset_index(), id_vars=['chromo', 'pos'], var_name='stat', value_name='value')
        return s

    def yza(self, fm_path, z_path, data_path, annos=None):
        # [chromo, pos, sample, y, z, anno]
        y = self.y(fm_path)
        z = self.z(z_path)
        a = self.a(data_path, annos)
        yza = pd.merge(pd.merge(y, z, how='inner'), a, how='inner')
        return yza

    def yzs(self ,fm_path, z_path, es_path, stats=None, nbins=3):
        # [chromo, pos, sample, y, z, stat, value, cut]
        y = self.y(fm_path)
        z = self.z(z_path)
        s = self.s(es_path, stats)
        yzs = pd.merge(pd.merge(y, z, how='inner'), s, how='inner').dropna()

        def group_cut(d):
            e = d.copy()
            if d.iloc[0].stat in ['cpg_cov']:
                f = pd.cut
            else:
                f = pd.qcut
            cuts, bins = f(e.value, nbins, retbins=True)
            e['cut'] = [str(x) for x in cuts]
            e.index = range(e.shape[0])
            return e

        if nbins:
            yzs = yzs.groupby('stat', group_keys=False).apply(group_cut)

        return yzs

