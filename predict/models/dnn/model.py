import yaml

import keras.models as kmodels
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
        self.lr_decay = 0.5
        self.max_epochs = 10
        self.early_stop = 3
        self.batch_size = 128
        self.shuffle = 'batch'

    @staticmethod
    def from_yaml(path):
        p = Params()
        with open(path, 'r') as f:
            t = yaml.load(f.read())
            p.update(t)
        return p

    def update(self, params):
        params = dict(params)
        self.__dict__.update(params)
        for k in ['seq', 'cpg', 'target']:
            if k in params.keys() and isinstance(params[k], dict):
                t = k.capitalize() + 'Params'
                vars(self)[k] = globals()[t]()
                vars(self)[k].update(params[k])

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
                            init='glorot_uniform',
                            activation=params.activation)
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
                            init='glorot_uniform',
                            activation='relu')
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

    model = kmodels.Graph()
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
