import pickle

from cpgkeras.models import CpgGraph
from cpgkeras.layers import core as kcore
from cpgkeras.layers import convolutional as kconv
from cpgkeras.layers import normalization as knorm
import cpgkeras.regularizers as kr
import cpgkeras.optimizers as kopt


def seq_layers(params):
    layers = []
    if params.drop_in:
        layer = kcore.Dropout(params.drop_in)
        layers.append(('xd', layer))
    nb_layer = len(params.nb_filter)
    w_reg = kr.WeightRegularizer(l1=params.l1, l2=params.l2)
    for l in range(nb_layer):
        layer = kconv.Convolution1D(nb_filter=params.nb_filter[l],
                                    filter_length=params.filter_len[l],
                                    activation=params.activation,
                                    init='glorot_uniform',
                                    W_regularizer=w_reg,
                                    border_mode='same')
        layers.append(('c%d' % (l + 1), layer))
        layer = kconv.MaxPooling1D(pool_length=params.pool_len[l])
        layers.append(('p%d' % (l + 1), layer))


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
                        activation=params.out_activation,
                        init='glorot_uniform')
    layers.append(('o', layer))
    return layers


def build(params, targets, seq_len, compile=True):
    model = CpgGraph()
    prev_nodes = []

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
        optimizer = optimizer_from_params(params)
        loss = {output: params.objective for output in outputs}
        model.compile(loss=loss, optimizer=optimizer)

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


def optimizer_from_config(config):
    optimizer_params = dict()
    for k, v in config.get('optimizer').items():
        optimizer_params[k] = v
    optimizer_name = optimizer_params.pop('name')
    optimizer = kopt.get(optimizer_name, optimizer_params)
    return optimizer


def optimizer_from_params(params):
    return kopt.get(params.optimizer, params.optimizer_params)


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
