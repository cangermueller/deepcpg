import keras.models as kmodels
from keras.layers import core as kcore
from keras.layers import convolutional as kconv
from keras.layers import normalization as knorm
import keras.optimizers as kopt


class CpgParams(object):

    def __init__(self):
        self.num_filters = 4
        self.filter_len = 4
        self.pool_len = 2
        self.num_hidden = 32
        self.dropout = 0.2
        self.dropout_inputs = 0.0
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
        self.num_filters = 4
        self.filter_len = 8
        self.pool_len = 4
        self.num_hidden = 32
        self.dropout = 0.2
        self.dropout_inputs = 0.0
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
        self.num_hidden = 16
        self.dropout = 0.2
        self.batch_norm = False

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

    def update(self, params):
        params = dict(params)
        for k in ['seq', 'cpg', 'target']:
            if k in params:
                vars(self)[k].update(params[k])
                del params[k]
        self.__dict__.update(params)

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
    if params.dropout_inputs:
        layer = kcore.Dropout(params.dropout_inputs)
        layers.append(('xd', layer))
    layer = kconv.Convolution1D(nb_filter=params.num_filters,
                                filter_length=params.filter_len,
                                activation='relu',
                                init='glorot_uniform',
                                border_mode='same')
    layers.append(('c1', layer))
    layer = kconv.MaxPooling1D(pool_length=params.pool_len)
    layers.append(('p1', layer))
    layer = kcore.Flatten()
    layers.append(('f1', layer))
    layer = kcore.Dropout(params.dropout)
    layers.append(('f1d', layer))
    if params.num_hidden:
        layer = kcore.Dense(params.num_hidden,
                            init='glorot_uniform')
        layers.append(('h1', layer))
        if params.batch_norm:
            layer = knorm.BatchNormalization()
            layers.append(('h1b', layer))
        layer = kcore.Activation('relu')
        layers.append(('h1a', layer))
        layer = kcore.Dropout(params.dropout)
        layers.append(('h1d', layer))
    return layers


def cpg_layers(params):
    layers = []
    if params.dropout_inputs:
        layer = kcore.Dropout(params.dropout_inputs)
        layers.append(('xd', layer))
    layer = kconv.Convolution2D(nb_filter=params.num_filters,
                                nb_row=1,
                                nb_col=nb_params.filter_len,
                                activation='relu',
                                init='glorot_uniform',
                                border_mode='same')
    layers.append(('c1', layer))
    layer = kconv.MaxPooling2D(poolsize=(1, params.pool_len))
    layers.append(('p1', layer))
    layer = kcore.Dropout(params.dropout)
    layers.append(('p1d', layer))
    if params.num_hidden:
        layer = kcore.Dense(params.num_hidden,
                            init='glorot_uniform')
        layers.append(('h1', layer))
        if params.batch_norm:
            layer = knorm.BatchNormalization()
            layers.append(('h1b', layer))
        layer = kcore.Activation('relu')
        layers.append(('h1a', layer))
        layer = kcore.Dropout(params.dropout)
        layers.append(('h1d', layer))
    return layers


def target_layers(params):
    layers = []
    if params.num_hidden:
        layer = kcore.Dense(params.num_hidden,
                            init='glorot_uniform')
        layers.append(('h1', layer))
        if params.batch_norm:
            layer = knorm.BatchNormalization()
            layers.append(('h1b', layer))
        layer = kcore.Activation('relu')
        layers.append(('h1a', layer))
        layer = kcore.Dropout(params.dropout)
        layers.append(('h1d', layer))
    layer = kcore.Dense(1,
                        activation='relu',
                        init='glorot_uniform')
    layers.append(('o', layer))
    return layers


def build(params, targets, seq_len=None, cpg_len=None, compile=True):
    model = kmodels.Graph()
    prev_nodes = []
    if params.seq:
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
        def label(x):
            return 'c_%s' % (x)

        layers = cpg_layers(params.seq)
        prev_node = label('x')
        model.add_input(prev_node, input_shape=(2, len(targets), cpg_len))
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
        import ipdb; ipdb.set_trace()
        if len(prev_nodes) > 1:
            model.add_node(inputs=prev_nodes, name=cur_node, layer=layer[1])
        else:
            model.add_node(input=prev_nodes[0], name=cur_node, layer=layer[1])
        layers = layers[1:]
        for layer in layers:
            cur_node = label(layer[0])
            model.add_node(input=prev_node, name=cur_node, layer=layer[1])
            prev_node = cur_node
        output = label('y')
        model.add_output(input=prev_node, name=output)
        outputs.append(output)

    if compile:
        opt = vars(kopt)[params.optimizer](**params.optimizer_args)
        loss = {output: 'binary_crossentropy' for output in outputs}
        model.compile(loss=loss, optimizer=opt)
