import numpy as np
import pandas as pd
import os.path as pt
#  import matplotlib
#  import matplotlib.pyplot as plt
#  import seaborn as sns

from predict import data_select
from predict import eval_stats
from predict import fm
from predict import predict as pred


#  matplotlib.style.use('ggplot')


class Loader(object):

    def __init__(self, group='test', chromos=None):
        self.chromos = chromos
        self.group = group

    def Y(self, fm_path):
        sel = fm.Selector(self.chromos)
        d = sel.select(fm_path, self.group, 'Y')
        return d

    def y(self, fm_path):
        # [chromo, pos, sample, y]
        y = self.Y(fm_path)
        y = pd.melt(y.reset_index(), id_vars=['chromo', 'pos'],
                    var_name='sample', value_name='y').dropna()
        return y

    def Z(self, z_path):
        d = pd.read_hdf(z_path, self.group)
        if self.chromos:
            d = d.loc[self.chromos]
        return d

    def z(self, z_path):
        # [chromo, pos, sample, z]
        z = self.Z(z_path)
        z = pd.melt(z.reset_index(), id_vars=['chromo', 'pos'],
                    var_name='sample', value_name='z')
        return z

    def A(self, data_path, annos=None, dist=False, group='/'):
        if annos is None:
            annos = True
        fsel = data_select.FeatureSelection()
        fsel.cpg = False
        fsel.knn = False
        fsel.knn_dist = False
        if dist:
            fsel.annos_dist = annos
        else:
            fsel.annos = annos
        sel = data_select.Selector(fsel)
        sel.chromos = self.chromos

        d = sel.select(data_path, group)
        assert len(d.columns.levels[0]) == 1
        d.columns = d.columns.droplevel(0)
        if dist:
            d = d == 0
        return d

    def a(self, data_path, annos=None, dist=False):
        # [chromo, pos, anno]
        a = self.A(data_path, annos=annos, dist=dist)
        a = pd.melt(a.reset_index(), id_vars=['chromo', 'pos'],
                    var_name='anno', value_name='is_in')
        a = a.assign(is_in=a.is_in == 1).query('is_in == True')
        a = a.loc[:, a.columns != 'is_in']
        return a

    def S(self, data_path, stats=None, group='es'):
        sel = eval_stats.Selector(self.chromos, stats)
        d = sel.select(data_path, group)
        return d

    def s(self, data_path, *args, **kwargs):
        # [chromo, pos, stat, value]
        s = self.S(data_path, *args, **kwargs)
        s = pd.melt(s.reset_index(), id_vars=['chromo', 'pos'],
                    var_name='stat', value_name='value')
        return s

    def yza(self, fm_path, z_path, data_path, annos=None, dist=False):
        # [chromo, pos, sample, y, z, anno]
        y = self.y(fm_path)
        z = self.z(z_path)
        a = self.a(data_path, annos=annos, dist=dist)
        yza = pd.merge(pd.merge(y, z, how='inner'), a, how='inner')
        return yza

    def yzs_group(self, yzs, nbins=3):
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
        yzs = yzs.groupby('stat', group_keys=False).apply(group_cut)
        return yzs

    def yzs(self, fm_path, z_path, data_path, stats=None, nbins=3):
        # [chromo, pos, sample, y, z, stat, value, cut]
        y = self.y(fm_path)
        z = self.z(z_path)
        s = self.s(data_path, stats=stats)
        yzs = pd.merge(pd.merge(y, z, how='inner'), s, how='inner').dropna()
        if nbins:
            yzs = self.yzs_group(yzs, nbins)
        return yzs


eval_annos = ['misc_Active_enhancers', 'misc_CGI', 'misc_CGI_shelf',
              'misc_CGI_shore', 'misc_Exons', 'misc_H3K27ac', 'misc_H3K27me3',
              'misc_H3K4me1', 'misc_H3K4me1_Tet1', 'misc_IAP',
              'misc_Intergenic', 'misc_Introns', 'misc_LMRs', 'misc_Oct4_2i',
              'misc_TSSs', 'misc_gene_body', 'misc_mESC_enhancers', 'misc_p300',
              'misc_prom_2k05k', 'misc_prom_2k05k_cgi', 'misc_prom_2k05k_ncgi',
              'rep_DNA', 'rep_LINE', 'rep_LTR', 'rep_SINE']

eval_funs = [('auc', pred.auc),
             ('acc', pred.acc),
             ('tpr', pred.tpr),
             ('tnr', pred.tnr),
             ('mcc', pred.mcc),
             ('rrmse', pred.rrmse),
             ('cor', pred.cor)]

def evaluate(y, z, mask=-1, funs=eval_funs):
    y = y.ravel()
    z = z.ravel()
    if mask is not None:
        t = y != mask
        y = y[t]
        z = z[t]
    s = dict()
    for name, fun in eval_funs:
        s[name] = fun(y, z)
    return pd.DataFrame(s, columns=[x for x, _ in eval_funs], index=[0])

def evaluate_all(y, z):
    keys = sorted(z.keys())
    p = [evaluate(y[k], z[k]) for k in keys]
    p = pd.concat(p)
    p.index = keys
    return p

def eval_to_str(e, index=False):
    s = e.to_csv(None, sep='\t', index=index, float_format='%.4f')
    return s


def get_weights(y, outputs):
    w = dict()
    for o in outputs:
        yo = y[o]
        wo = np.zeros(len(yo), dtype='float32')
        wo[yo != -1] = 1
        w[o] = wo
    return w

def read_data(path, max_samples=None):
    f = h5.File(path, 'r')
    data = dict()
    for k, v in f.items():
        if max_samples is None:
            data[k] = v.value
        else:
            data[k] = v[:max_samples]
    c = data['c_x'].astype('float32')
    c[:, 1] /= c[:, 1].max()
    data['c_x'] = c
    f.close()
    return data

class App(object):

    def run(self, args):
        name = pt.basename(args[0])
        parser = self.create_parser(name)
        opts = parser.parse_args(args[1:])
        return self.main(name, opts)

    def create_parser(self, name):
        p = argparse.ArgumentParser(
            prog=name,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Build and train model')
        p.add_argument(
            '--train_data',
            help='Training data file')
        p.add_argument(
            '--val_data',
            help='Validation data file')
        p.add_argument(
            '--test_data',
            help='Test data file')
        p.add_argument(
            '-o', '--out_dir',
            help='Output directory')
        p.add_argument(
            '--model_params',
            help='Model parameters file')
        p.add_argument(
            '--model_file',
            help='JSON model description')
        p.add_argument(
            '--model_weights',
            help='Model weights')
        p.add_argument(
            '--max_samples',
            help='Limit # samples',
            type=int)
        p.add_argument(
            '--seed',
            help='Seed of rng',
            type=int,
            default=0)
        p.add_argument(
            '--verbose',
            help='More detailed log messages',
            action='store_true')
        p.add_argument(
            '--log_file',
            help='Write log messages to file')
        return p

    def main(self, name, opts):
        logging.basicConfig(filename=opts.log_file,
                            format='%(levelname)s (%(asctime)s): %(message)s')
        log = logging.getLogger(name)
        if opts.verbose:
            log.setLevel(logging.DEBUG)
        else:
            log.setLevel(logging.INFO)
            log.debug(opts)


        if opts.seed is not None:
            np.random.seed(opts.seed)
        pd.set_option('display.width', 150)

        log.info('Read data')
        train_data = read_data(opts.train_data, opts.max_samples)
        #  train_data = filter(lambda x: not re.match('u[1-9]+_y', x[0]), train_data.items())

        model_params = NetParams()
        if opts.model_params:
            with open(opts.model_params, 'r') as f:
                configs = yaml.load(f.read())
                model_params.update(configs)
        print('Model parameters:')
        print(model_params)
        print()

        if opts.model_file:
            log.info('Build model from file')
            with open(opts.model_file, 'r') as f:
                model = f.read()
            model = kmodels.model_from_json(model)
        else:
            log.info('Build model')
            model = build_model(model_params)
            with open(pt.join(opts.out_dir, 'model.json'), 'w') as f:
                f.write(model.to_json())

        if opts.train_data is None:
            return 0

        print('%d training samples' % (len(train_data['pos'])))
        if opts.val_data:
            val_data = read_data(opts.val_data)
            print('%d validation samples' % (len(val_data['pos'])))
        else:
            val_data = train_data

        sample_weight_train = get_weights(train_data, model.output_order)
        sample_weight_val = get_weights(val_data, model.output_order)

        log.info('Fit model')
        cb = []
        cb.append(ModelCheckpoint(pt.join(opts.out_dir, 'model_weights')))
        cb.append(EarlyStopping(patience=model_params.early_stop, verbose=1))
        pl = PerformanceLogger(train_data, val_data)
        cb.append(pl)
        model.fit(train_data, validation_data=val_data,
                  callbacks=cb, nb_epoch=model_params.max_epochs,
                  verbose=2,
                  sample_weight=sample_weight_train,
                  sample_weight_val=sample_weight_val)

        t = pt.join(opts.out_dir, 'model_weights_best.h5')
        if pt.isfile(t):
            model.load_weights(t)

        for n in ['train', 'val']:
            pl.logs[n].to_csv(pt.join(opts.out_dir, 'perf_%s.csv' % (n)), sep='\t', float_format='%.4f')

        print('\nTraining set performance:')
        z_train = model.predict(train_data)
        print(evaluate_all(train_data, z_train))

        print('\nValidation set loss:')
        z_val = model.predict(val_data)
        print(evaluate_all(val_data, z_val))

        if opts.test_data is not None:
            test_data = read_data(opts.test_data)
            print('\nTest set performance:')
            z_test = model.predict(test_data)
            print(evaluate_all(test_data, z_test))

        log.info('Done!')

        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)

#  def plot_annos(pa):
    #  pam = pa.groupby('anno').apply(lambda x: pd.DataFrame(dict(mean=x.auc.mean()),
                                                          #  index=[0])).reset_index(level=0)
    #  pam.sort('mean', ascending=False, inplace=True)
    #  fig, ax = plt.subplots(figsize=(10, len(pa.anno.unique()) * 0.5))
    #  sns.boxplot(y='anno', x='auc', data=pa, orient='h', order=pam.anno, ax=ax)
    #  sns.stripplot(y='anno', x='auc', data=pa, orient='h', order=pam.anno,
                  #  jitter=True, size=5, color='black', edgecolor='black', ax=ax)
    #  return (fig, ax)


#  def plot_stats(ps):
    #  grid = sns.FacetGrid(ps, col='stat', hue='stat', col_wrap=2,
                         #  sharex=False, size=6)
    #  grid.map(plt.plot, 'mean', 'auc', marker="o", ms=6, linewidth=2)
    #  return grid


class Evaluater(object):

    def __init__(self, data_path, fm_path, base_path='.', group='test'):
        self.data_path = data_path
        self.fm_path = fm_path
        self.base_path = base_path
        self.group = group
        self.loader = Loader(group)
        self.__y = None
        self.__z = None
        self.__yz = None
        self.logger = None

    def log(self, x):
        if (self.logger):
            self.logger(x)

    def get_data(self, dset, weight=False):
        X, Y, w = self.data[dset]
        if weight:
            return (X, Y, w)
        else:
            return (X, Y)

    def __load_y(self):
        if self.__y is not None:
            return self.__y
        self.__y = self.loader.y(self.fm_path)
        return self.__y

    def __load_z(self):
        if self.__z is not None:
            return self.__z
        z_path = pt.join(self.base_path, 'z.h5')
        self.__z = self.loader.z(z_path)
        return self.__z

    def __load_yz(self):
        if self.__yz is not None:
            return self.__yz
        y = self.__load_y()
        z = self.__load_z()
        self.__yz = pd.merge(y, z, how='inner')
        return self.__yz

    def eval_annos(self):
        self.log('Evaluate annotations ...')

        def group_annos(d):
            scores = dict()
            cols = []
            for name, fun in eval_funs:
                if name == 'auc' and len(d.y.unique()) == 1:
                    v = np.nan
                else:
                    v = fun(d.y, d.z)
                scores[name] = v
                cols.append(name)
            s = pd.DataFrame(scores, columns=cols, index=[0])
            return s

        yz = self.__load_yz()
        a = self.loader.a(self.data_path, annos=eval_annos, dist=True)
        yza = pd.merge(yz, a, how='inner')

        pa = yza.groupby(['anno', 'sample']).apply(group_annos)
        pa.index = pa.index.droplevel(2)
        pa.reset_index(inplace=True)
        pa.dropna(inplace=True)
        pa.to_csv(pt.join(self.base_path, 'perf_annos.csv'),
                  sep='\t', index=False)
        fig, ax = plot_annos(pa)
        fig.savefig(pt.join(self.base_path, 'perf_annos.pdf'))

        return pa

    def eval_stats(self, nbins=4, stats=None):
        self.log('Evaluate statistics ...')

        def group_stats(d):
            scores = dict()
            cols = []
            for name, fun in eval_funs:
                if name == 'auc' and len(d.y.unique()) == 1:
                    v = np.nan
                else:
                    v = fun(d.y, d.z)
                scores[name] = v
                cols.append(name)
            scores['mean'] = d.value.mean()
            cols.append('mean')
            s = pd.DataFrame(scores, columns=cols, index=[0])
            return s

        yz = self.__load_yz()
        s = self.loader.s(self.data_path, stats=stats)
        yzs = pd.merge(yz, s, how='inner').dropna()
        if nbins:
            yzs = self.loader.yzs_group(yzs, nbins)
        yzs = yzs.loc[yzs.stat != 'cpg_cov']

        ps = yzs.groupby(['stat', 'cut']).apply(group_stats).reset_index(level=(0, 1))
        ps.sort(['stat', 'mean'], inplace=True)
        ps.to_csv(pt.join(self.base_path, 'perf_stats.csv'),
                  sep='\t', index=False)
        g = plot_stats(ps)
        g.savefig(pt.join(self.base_path, 'perf_stats.pdf'))

        return ps

    def run(self):
        self.eval_annos()
        self.eval_stats()
