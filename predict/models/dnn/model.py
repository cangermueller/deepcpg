import yaml
import numpy as np
import re
import pickle
import os.path as pt

from keras.models import CpgGraph
from keras.layers import core as kcore
from keras.layers import convolutional as kconv
from keras.layers import normalization as knorm
import keras.optimizers as kopt


class CpgParams(object):

    def __init__(self):
        self.activation = 'relu'
        self.nb_filter = 4
        self.filter_len = 4
        self.pool_len = 2
        self.nb_hidden = 32
        self.drop_in = 0.0
        self.drop_out = 0.2
        self.batch_norm = False

    def validate(self):
        self.pool_len = min(self.pool_len, self.nb_filter)

    def update(self, params):
        self.__dict__.update(params)

    def __str__(self):
        params = vars(self)
        s = ''
        for k in sorted(params.keys()):
            s += '%s: %s\n' % (k, str(params[k]))
        return s.strip()


class SeqParams(object):

    def __init__(self):
        self.activation = 'relu'
        self.nb_filter = 4
        self.filter_len = 8
        self.pool_len = 4
        self.nb_hidden = 32
        self.drop_in = 0.0
        self.drop_out = 0.2
        self.batch_norm = False

    def validate(self):
        self.pool_len = min(self.pool_len, self.nb_filter)

    def update(self, params):
        self.__dict__.update(params)

    def __str__(self):
        params = vars(self)
        s = ''
        for k in sorted(params.keys()):
            s += '%s: %s\n' % (k, str(params[k]))
        return s.strip()


class TargetParams(object):

    def __init__(self):
        self.activation = 'relu'
        self.nb_hidden = 16
        self.drop_out = 0.2
        self.batch_norm = False

    def update(self, params):
        self.__dict__.update(params)

    def __str__(self):
        params = vars(self)
        s = ''
        for k in sorted(params.keys()):
            s += '%s: %s\n' % (k, str(params[k]))
        return s.strip()


class Params(object):

    def __init__(self):
        self.seq = SeqParams()
        self.cpg = CpgParams()
        self.target = TargetParams()

        self.optimizer = 'Adam'
        self.optimizer_params = {'lr': 0.001}

    def validate(self):
        for k in ['seq', 'cpg', 'target']:
            if hasattr(vars(self)[k], 'validate'):
                vars(self)[k].validate()
        t = self.target.nb_hidden
        for k in ['seq', 'cpg']:
            s = vars(self)[k]
            if hasattr(s, 'nb_hidden'):
                t = min(t, s.nb_hidden)
        self.target.nb_hidden = t

    @staticmethod
    def from_yaml(path):
        p = Params()
        with open(path, 'r') as f:
            t = yaml.load(f.read())
            p.update(t)
        return p

    def to_yaml(self, path):
        dparam = self.__dict__
        with open(path, 'w') as f:
            t = yaml.dump(dparam, default_flow_style=False)
            t = re.subn('!![^\s]+', '', t)[0]
            f.write(t)

    def update(self, params):
        vself = vars(self)
        for k, v in dict(params).items():
            if k in ['seq', 'cpg', 'target']:
                if isinstance(v, dict):
                    t = k.capitalize() + 'Params'
                    vself[k] = globals()[t]()
                    vself[k].update(params[k])
                else:
                    vself[k] = v
            elif k in vself.keys():
                vself[k] = v

    def __str__(self):
        s = 'Seq model:\n'
        s += '---------\n'
        s += str(self.seq)
        s += '\n\nCpG model:\n'
        s += '----------\n'
        s += str(self.cpg)
        s += '\n\nTarget model:\n'
        s += '-------------\n'
        s += str(self.target)
        s += '\n'

        params = vars(self)
        for k in sorted(params.keys()):
            if k not in ['seq', 'cpg', 'target']:
                s += '\n%s: %s' % (k, params[k])
        return s


def sample_dict(param_dist):
    sample = dict()
    for k, v in param_dist.items():
        if isinstance(v, dict):
            sample[k] = sample_dict(v)
        elif isinstance(v, list):
            sample[k] = v[np.random.randint(0, len(v))]
        elif hasattr(v, 'rvs'):
            sample[k] = v.rvs()
        else:
            sample[k] = v
    return sample


class ParamSampler(object):

    def __init__(self, param_dist, nb_sample=1,
                 global_param=['batch_norm', 'activation']):
        self.param_dist = param_dist
        self.nb_sample = nb_sample
        self.global_param = global_param

    def __iter__(self):
        self._nb_sample = 0
        return self

    def __next__(self):
        if self._nb_sample == self.nb_sample:
            raise StopIteration

        sample = sample_dict(self.param_dist)
        gparam = dict()
        t = dict()
        for k, v in sample.items():
            if k in self.global_param:
                gparam[k] = v
            else:
                t[k] = v
        sample = t

        param = Params()
        param.update(sample)
        if 'batch_norm' in gparam and gparam['batch_norm']:
            param.optimizer = 'sgd'
        for k, v in gparam.items():
            for s in ['cpg', 'seq', 'target']:
                sub = vars(param)[s]
                if hasattr(sub, '__dict__') and k in vars(sub):
                    vars(sub)[k] = v
        param.validate()

        self._nb_sample += 1
        return param


def standardize_weights(y, sample_weight=None, class_weight=None):
    if sample_weight is not None:
        return sample_weight
    elif isinstance(class_weight, dict):
        if len(y.shape) > 3:
            raise Exception('class_weight not supported for 4+ dimensional targets.')
        yshape = y.shape
        # for time-distributed data, collapse time and sample
        y = np.reshape(y, (-1, yshape[-1]))
        if y.shape[1] > 1:
            y_classes = y.argmax(axis=1)
        elif y.shape[1] == 1:
            y_classes = np.reshape(y, y.shape[0])
        else:
            y_classes = y
        class_weights = np.asarray([class_weight[cls] for cls in y_classes])
        return np.reshape(class_weights, yshape[:-1] + (1,))  # uncollapse initial dimensions
    else:
        return np.ones(y.shape[:-1] + (1,))


def cpg_layers(params):
    layers = []
    if params.drop_in:
        layer = kcore.Dropout(params.drop_in)
        layers.append(('xd', layer))
    layer = kconv.Convolution2D(nb_filter=params.nb_filter,
                                nb_row=1,
                                nb_col=params.filter_len,
                                activation=params.activation,
                                init='glorot_uniform',
                                border_mode='same')
    layers.append(('c1', layer))
    layer = kconv.MaxPooling2D(pool_size=(1, params.pool_len))
    layers.append(('p1', layer))
    layer = kcore.Flatten()
    layers.append(('f1', layer))
    if params.drop_out:
        layer = kcore.Dropout(params.drop_out)
        layers.append(('f1d', layer))
    if params.nb_hidden:
        layer = kcore.Dense(params.nb_hidden,
                            activation='linear',
                            init='glorot_uniform')
        layers.append(('h1', layer))
        if params.batch_norm:
            layer = knorm.BatchNormalization()
            layers.append(('h1b', layer))
        layer = kcore.Activation(params.activation)
        layers.append(('h1a', layer))
        if params.drop_out:
            layer = kcore.Dropout(params.drop_out)
            layers.append(('h1d', layer))
    return layers


def seq_layers(params):
    layers = []
    if params.drop_in:
        layer = kcore.Dropout(params.drop_in)
        layers.append(('xd', layer))
    layer = kconv.Convolution1D(nb_filter=params.nb_filter,
                                filter_length=params.filter_len,
                                activation=params.activation,
                                init='glorot_uniform',
                                border_mode='same')
    layers.append(('c1', layer))
    layer = kconv.MaxPooling1D(pool_length=params.pool_len)
    layers.append(('p1', layer))
    layer = kcore.Flatten()
    layers.append(('f1', layer))
    if params.drop_out:
        layer = kcore.Dropout(params.drop_out)
        layers.append(('f1d', layer))
    if params.nb_hidden:
        layer = kcore.Dense(output_dim=params.nb_hidden,
                            activation='linear',
                            init='glorot_uniform')
        layers.append(('h1', layer))
        if params.batch_norm:
            layer = knorm.BatchNormalization()
            layers.append(('h1b', layer))
        layer = kcore.Activation(params.activation)
        layers.append(('h1a', layer))
        if params.drop_out:
            layer = kcore.Dropout(params.drop_out)
            layers.append(('h1d', layer))
    return layers


def target_layers(params):
    layers = []
    if params.nb_hidden:
        layer = kcore.Dense(params.nb_hidden,
                            activation='linear',
                            init='glorot_uniform')
        layers.append(('h1', layer))
        if params.batch_norm:
            layer = knorm.BatchNormalization()
            layers.append(('h1b', layer))
        layer = kcore.Activation(params.activation)
        layers.append(('h1a', layer))
        if params.drop_out:
            layer = kcore.Dropout(params.drop_out)
            layers.append(('h1d', layer))
    layer = kcore.Dense(1,
                        activation='sigmoid',
                        init='glorot_uniform')
    layers.append(('o', layer))
    return layers


def build(params, targets, seq_len=None, cpg_len=None, compile=True,
          nb_unit=None):
    if nb_unit is None:
        nb_unit = len(targets)

    model = CpgGraph()
    prev_nodes = []
    if params.seq:
        assert seq_len is not None, 'seq_len required!'
        def label(x):
            return 's_%s' % (x)

        layers = seq_layers(params.seq)
        prev_node = label('x')
        model.add_input(prev_node, input_shape=(seq_len, 4))
        for layer in layers:
            cur_node = label(layer[0])
            model.add_node(input=prev_node, name=cur_node, layer=layer[1])
            prev_node = cur_node
        prev_nodes.append(prev_node)

    if params.cpg:
        assert cpg_len is not None, 'cpg_len required!'
        def label(x):
            return 'c_%s' % (x)

        layers = cpg_layers(params.cpg)
        prev_node = label('x')
        model.add_input(prev_node, input_shape=(2, nb_unit, cpg_len))
        for layer in layers:
            cur_node = label(layer[0])
            model.add_node(input=prev_node, name=cur_node, layer=layer[1])
            prev_node = cur_node
        prev_nodes.append(prev_node)

    outputs = []
    for target in targets:
        def label(x):
            return '%s_%s' % (target, x)

        layers = target_layers(params.target)
        layer = layers[0]
        cur_node = label(layer[0])
        if len(prev_nodes) > 1:
            model.add_node(inputs=prev_nodes, name=cur_node, layer=layer[1])
        else:
            model.add_node(input=prev_nodes[0], name=cur_node, layer=layer[1])
        prev_node = cur_node
        layers = layers[1:]
        for layer in layers:
            cur_node = label(layer[0])
            model.add_node(input=prev_node, name=cur_node, layer=layer[1])
            prev_node = cur_node
        output = label('y')
        model.add_output(input=prev_node, name=output)
        outputs.append(output)

    if compile:
        opt = vars(kopt)[params.optimizer](**params.optimizer_params)
        loss = {output: 'binary_crossentropy' for output in outputs}
        model.compile(loss=loss, optimizer=opt)

    return model


def model_from_json(json_file, weights_file=None, compile=True):
    import keras.models as kmodels
    with open(json_file, 'r') as f:
        model = f.read()
    model = kmodels.model_from_json(model, compile=compile)
    model.load_weights(weights_file)
    return model


def model_from_pickle(pickle_file):
    with open(pickle_file, 'rb') as f:
        model = pickle.load(f)
    return model


def model_from_list(fnames, *args, **kwargs):
    if not isinstance(fnames, list):
        fnames = list(fnames)
    if len(fnames) == 2:
        model = model_from_json(fnames[0], fnames[1], *args, **kwargs)
    else:
        model = model_from_pickle(fnames[0])
    return model


def copy_weights(src, dst, prefix):
    n = 0
    for k, v in src.nodes.items():
        if k.startswith(prefix) and k in dst.nodes:
            dst.nodes[k].set_weights(src.nodes[k].get_weights())
            n += 1
    return n


def model_to_pickle(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model, f)


def model_to_json(model, json_file, weights_file=None):
    with open(json_file, 'w') as f:
        f.write(model.to_json())
    if weights_file is not None:
        model.save_weights(weights_file, overwrite=True)
