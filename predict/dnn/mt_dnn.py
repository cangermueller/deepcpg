import numpy as np
import pandas as pd
import theano as th
import theano.tensor as tht

import predict.dnn.utils as dut
import predict.predict as pred


class DnnLayer(object):

    def __init__(self, X, nin, nout, act=dut.relu, W=None, b=None, rng=0):

        if W is None:
            winit = dut.WeightsInitializer(nin, nout, rng)
            if act == dut.relu:
                f = winit.relu
            elif act == dut.sigmoid:
                f = winit.sigmoid
            elif act == dut.tanh:
                f = winit.tanh
            else:
                f = winit.uniform()
            W = th.shared(f(), 'W', borrow=True)
        if b is None:
            b = th.shared(np.zeros(nout), 'b', borrow=True)

        Z = X.dot(W) + b
        Y = act(Z)
        self.Y = Y
        self.W = W
        self.b = b
        self.params = [W, b]
        self.nin = nin
        self.nout = nout


class MtDnnModel(object):

    def __dropout(self, X, rate):
        Y = X * self.rns.binomial((X.shape), 1, 1 - rate)
        return Y

    def __init__(self, X, nin, nhidden=[], nout=1, ntasks=1,
                 act_hidden=dut.relu, act_out=dut.sigmoid,
                 drop_in=0.0, drop_hidden=0.0, rng=0, rns=None):
        if isinstance(rng, int):
            rng = np.random.RandomState(rng)
        self.rng = rng
        if rns is None:
            rns = tht.shared_randomstreams.RandomStreams(self.rng.get_state()[1][0])
        self.rns = rns
        self.drop_in = th.shared(drop_in, 'drop_in')
        self.drop_hidden = th.shared(drop_hidden, 'drop_hidden')

        if nhidden is None:
            nhidden = []
        if not isinstance(nhidden, list):
            nhidden = [nhidden]

        acts = [act_hidden] * len(nhidden) + [act_out]
        nunits = nhidden + [nout]
        nlayers = len(nunits)
        hidden_layers = []
        output_layers = []
        for i in range(nlayers):
            if i == 0:
                layer_X = X
                if drop_in > 0:
                    layer_X = self.__dropout(layer_X, self.drop_in)
                layer_nin = nin
            else:
                layer_X = hidden_layers[i - 1].Y
                if drop_hidden > 0:
                    layer_X = self.__dropout(layer_X, self.drop_hidden)
                layer_nin = hidden_layers[i - 1].nout
            if i == nlayers - 1:
                for k in range(ntasks):
                    layer = DnnLayer(layer_X, layer_nin, nunits[i], acts[i],
                                     rng=self.rng)
                    output_layers.append(layer)
            else:
                layer = DnnLayer(layer_X, layer_nin, nunits[i], acts[i],
                                 rng=self.rng)
                hidden_layers.append(layer)

        hidden_params = []
        for layer in hidden_layers:
            hidden_params.extend([layer.W, layer.b])
        params = []
        for layer in output_layers:
            params.append(list(hidden_params) + [layer.W, layer.b])

        self.X = X
        self.Y = [output_layers[k].Y for k in range(ntasks)]
        self.nin = nin
        self.nout = nout
        self.hidden_layers = hidden_layers
        self.output_layers = output_layers
        self.hidden_params = hidden_params
        self.params = params

    def set_dropout(self, drop_in=0.0, drop_hidden=0.0):
        self.drop_in.set_value(drop_in)
        self.drop_hidden.set_value(drop_hidden)

    def get_droupout(self):
        t = (self.drop_in.get_value(), self.drop_hidden.get_value())
        t = [float(t) for t in t]
        return t



class MtDnnParams(object):

    def __init__(self):
        # Architecture
        self.nhidden = []
        self.hidden_act = dut.relu
        # Training
        self.drop_in = 0.1
        self.drop_hidden = 0.5
        self.l1_reg = 0.0
        self.l2_reg = 0.0
        self.mom = 0.9
        # float or list
        self.lr_init = 0.01
        self.lr_final = 10**-6
        # less than 5% improvement on validation set
        self.lr_decay_thr = None
        self.max_epochs = 30
        self.batch_size = 1000
        # True or number of epochs without improvement
        self.early_stop = False
        # Task-specific class weights
        self.wc = None

    def __str__(self):
        return str(self.__dict__)


class MtDnn(object):

    def __init__(self, params=None, rng=0, logger=None):
        if params is None:
            params = MtDnnParams()
        self.params = params
        self.rng = rng
        self.logger = logger
        self.skip_epoch = 1
        self.log_auc_train = False
        self.log_auc_val = True

    def log(self, x):
        if self.logger:
            self.logger(x)

    def fit(self, train_X, train_Y, val_X=None, val_Y=None,
            train_ws=None, val_ws=None, lc_logger=None):

        def prepro(X, Y, ws):
            X, Y = format_XY(X, Y)
            if ws is None:
                ws = np.ones(train_X.shape[0], dtype=np.float32)
            else:
                ws = np.asarray(train_ws, dtype=np.float32)
            return (X, Y, ws)

        train_X, train_Y, train_ws = prepro(train_X, train_Y, train_ws)
        if val_X is None:
            val_X, val_Y, val_ws = train_X, train_Y, train_ws
        else:
            val_X, val_Y, val_ws = prepro(val_X, val_Y, val_ws)
        self.ntasks = len(train_Y)
        self.lc_logger = lc_logger

        # task-specific class weights
        wc = self.params.wc
        if wc is None:
            wc = np.ones(2)
        if not isinstance(wc, list):
            wc = [wc] * self.ntasks
        for i in range(len(wc)):
            t = np.asarray(wc[i], dtype=np.float32)
            t *= 2 / t.sum()
            wc[i] = th.shared(t, borrow=True)
        self.wc = wc

        self.nin = train_X.shape[1]
        self.setup()

        self.__cost_train_prev = None
        self.__cost_val_prev = None
        self.__lr_decay = None
        self.__nearly_stop = 0
        self.train_X = train_X
        self.train_Y = train_Y
        self.train_ws = train_ws
        self.val_X = val_X
        self.val_Y = val_Y
        self.val_ws = val_ws

        sgd = dut.Sgd(batch_size=self.params.batch_size,
                  max_epochs=self.params.max_epochs,
                  shuffle=True, rng=self.rng)
        sgd.optimize(train_X.shape[0], self.sgd_train_fun, self.sgd_epoch_fun);
        self.mlp.set_dropout()

        self.train_X = None
        self.train_Y = None
        self.train_ws = None
        self.val_X = None
        self.val_Y = None
        self.val_ws = None
        self.lc_logger = None


    def setup(self):
        params = self.params
        nout = 1

        X = tht.dmatrix('X')
        m = MtDnnModel(X, self.nin, nhidden=params.nhidden, nout=nout,
                       ntasks=self.ntasks,
                       drop_in=params.drop_in, drop_hidden=params.drop_hidden,
                       rng=self.rng)
        # sample weights
        ws = tht.fvector('ws')

        cost_funs = []
        train_funs = []
        predict_funs = []
        lrs = []
        costs = []
        self.dparams = []
        for task in range(self.ntasks):
            Y = tht.fmatrix('Y')
            # cost
            cost = dut.wnll(Y, m.Y[task], self.wc[task], ws)
            cost /= Y.shape[0]
            if params.l1_reg:
                for layer in m.layers:
                    cost += params.l1_reg * tht.sum(tht.abs_(layer.W))
            if params.l2_reg:
                for layer in m.layers:
                    cost += params.l2_reg * tht.sum(layer.W**2)

            dparams = tht.grad(cost, m.params[task])
            ug = dut.UpdatesGenerator(m.params[task], dparams)
            # task specific learning rate
            if isinstance(params.lr_init, list):
                lr_init = params.lr_init[task]
            else:
                lr_init = params.lr_init
            lr = th.shared(lr_init, 'lr')
            updates = ug.momentum(lr=lr, mom=params.mom)
            lrs.append(lr)
            train_fun = th.function([X, Y, ws], cost, updates=updates)
            train_funs.append(train_fun)
            cost_fun = th.function([X, Y, ws], cost)
            cost_funs.append(cost_fun)
            predict_fun = th.function([X], m.Y[task])
            predict_funs.append(predict_fun)
            costs.append(cost)
            self.dparams.append(dparams)

        self.mlp = m
        self.predict_funs = predict_funs
        self.cost_funs = cost_funs
        self.train_funs = train_funs
        self.lrs = lrs

    def sgd_train_fun(self, idx):
        X_batch = self.train_X[idx]
        ws_batch = self.train_ws[idx]
        for task in range(self.ntasks):
            Y_task = self.train_Y[task][idx]
            t = ~np.isnan(Y_task).ravel()
            if np.sum(t) > 0:
                X_task = X_batch[t]
                Y_task = Y_task[t]
                ws_task = ws_batch[t]
                cost = self.train_funs[task](X_task, Y_task, ws_task)
                if np.isnan(cost) or np.isinf(cost):
                    raise RuntimeError('Invalid training cost!')

    def sgd_epoch_fun(self, epoch):
        params = self.params
        if (epoch % self.skip_epoch) != 0:
            return

        def format_header(columns):
            t = ['%10s' % (x) for x in columns]
            t.insert(0, ' T')
            return '\t'.join(t)

        def format_line(target, values):
            t = ['%10.3f' % (x) for x in values]
            t.insert(0, '%2d' % (target))
            return '\t'.join(t)

        costs_train = []
        costs_val = []
        aucs_train = []
        aucs_val = []
        lrs = []
        for k in range(self.ntasks):
            # train
            X, Y, ws = pred.complete_array(self.train_X, self.train_Y[k],
                                           self.train_ws)
            cost_train = self.cost_funs[k](X, Y, ws)
            auc_train = np.nan
            if self.log_auc_train:
                auc_train = pred.auc(Y, self.predict_funs[k](X))
            costs_train.append(cost_train)
            aucs_train.append(auc_train)
            # val
            if self.val_X is self.train_X:
                costs_val.append(cost_train)
                aucs_val.append(auc_train)
            else:
                X, Y, ws = pred.complete_array(self.val_X, self.val_Y[k],
                                            self.val_ws)
                cost_val = self.cost_funs[k](X, Y, ws)
                costs_val.append(cost_val)
                auc_val = np.nan
                if self.log_auc_val:
                    auc_val = pred.auc(Y, self.predict_funs[k](X))
                aucs_val.append(auc_val)
            lrs.append(self.lrs[k].get_value())

        stats = pd.DataFrame(
            {'cost_train': costs_train,
             'cost_val': costs_val,
             'auc_train': aucs_train,
             'auc_val': aucs_val,
             'lr': lrs
             },
            columns=['cost_train', 'cost_val', 'auc_train', 'auc_val', 'lr']
            )
        if self.lc_logger is not None:
            self.lc_logger(stats)
        self.log('-' * 100)
        self.log('Epoch: %2d' % (epoch))
        self.log(format_header(stats.columns))
        for k in range(stats.shape[0]):
            self.log(format_line(k + 1, stats.iloc[k].values))
        self.log(format_line(0, stats.mean().values))

        # Early stopping
        params = self.params
        cost_val = stats.cost_val.mean()
        if self.__cost_val_prev is not None:
            if params.early_stop:
                if self.__cost_val_prev < cost_val:
                    self.log('%d' % self.__nearly_stop)
                    self.__nearly_stop += 1
                    if isinstance(params.early_stop, bool) or \
                        self.__nearly_stop == params.early_stop:
                        self.log('Early stop!')
                        return True
                else:
                    self.__nearly_stop = 0
        self.__cost_val_prev = cost_val

        # Learning rate decay
        cost_train = stats.cost_train.mean()
        if params.lr_decay_thr is not None:
            if self.__lr_decay is None and self.__cost_train_prev is not None:
                improve = (self.__cost_train_prev - cost_train) / self.__cost_train_prev
                if improve <= params.lr_decay_thr:
                    self.__lr_decay = []
                    for k in range(self.ntasks):
                        self.__lr_decay.append((self.lrs[k].get_value() - params.lr_final) / (params.max_epochs - epoch))
                        self.log('Learning rate decay activated for task %d: %.3f' % (k + 1, self.__lr_decay[k]))
                        if self.__lr_decay is not None:
                            for k in range(self.ntasks):
                                self.lrs[k].set_value(self.lrs[k].get_value() - self.__lr_decay[k])
                                self.__cost_train_prev = cost_train

    def predict(self, X, task=None):
        if task is not None:
            return self.predict_funs[task](X)
        else:
            z = []
            for pf in self.predict_funs:
                z.append(pf(X))
            return np.hstack(z)


def format_X(X):
    return np.asarray(X).astype('float32')


def format_Y(Y):
    if isinstance(Y, pd.DataFrame):
        Y = [Y.iloc[:, i].values for i in range(Y.shape[1])]
    if not isinstance(Y, list):
        Y = [Y]
    for i in range(len(Y)):
        y = Y[i]
        y = np.asarray(y).astype('float16')

        if y.ndim == 1:
            y = y.reshape(-1, 1)
        Y[i] = y
    return Y


def format_XY(X, Y):
    X = format_X(X)
    Y = format_Y(Y)
    return (X, Y)
