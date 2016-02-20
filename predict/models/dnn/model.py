import pickle

from keras.models import CpgGraph
from keras.layers import core as kcore
from keras.layers import convolutional as kconv
from keras.layers import normalization as knorm
import keras.regularizers as kr
import keras.optimizers as kopt


def cpg_layers(params):
    layers = []
    if params.drop_in:
        layer = kcore.Dropout(params.drop_in)
        layers.append(('xd', layer))
    nb_layer = len(params.nb_filter)
    w_reg = kr.WeightRegularizer(l1=params.l1, l2=params.l2)
    for l in range(nb_layer):
        layer = kconv.Convolution2D(nb_filter=params.nb_filter[l],
                                    nb_row=1,
                                    nb_col=params.filter_len[l],
                                    activation=params.activation,
                                    init='glorot_uniform',
                                    W_regularizer=w_reg,
                                    border_mode='same')
        layers.append(('c%d' % (l + 1), layer))
        layer = kconv.MaxPooling2D(pool_size=(1, params.pool_len[l]))
        layers.append(('p%d' % (l + 1), layer))

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


def joint_layers(params):
    layers = []
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


def add_layers(model, layers, prev_nodes, prefix):
    for layer in layers:
        cur_node, cur_layer = layer
        cur_node = '%s_%s' % (prefix, cur_node)
        if len(prev_nodes) > 1:
            model.add_node(inputs=prev_nodes, name=cur_node, layer=cur_layer)
        else:
            model.add_node(input=prev_nodes[0], name=cur_node, layer=cur_layer)
        prev_nodes = [cur_node]
    return prev_nodes


def build(params, targets, seq_len=None, cpg_len=None, compile=True,
          nb_unit=None):

    model = CpgGraph()
    branch_nodes = []
    if params.seq:
        assert seq_len is not None, 'seq_len required!'
        prev_nodes = ['s_x']
        model.add_input(prev_nodes[0], input_shape=(seq_len, 4))
        layers = seq_layers(params.seq)
        prev_nodes = add_layers(model, layers, prev_nodes, 's')
        branch_nodes.extend(prev_nodes)

    if params.cpg:
        assert cpg_len is not None, 'cpg_len required!'
        if nb_unit is None:
            nb_unit = len(targets)
        prev_nodes = ['c_x']
        model.add_input(prev_nodes[0], input_shape=(2, nb_unit, cpg_len))
        layers = cpg_layers(params.cpg)
        prev_nodes = add_layers(model, layers, prev_nodes, 'c')
        branch_nodes.extend(prev_nodes)

    if params.joint and params.joint.nb_hidden > 0:
        layers = joint_layers(params.joint)
        branch_nodes = add_layers(model, layers, branch_nodes, 'j')

    outputs = []
    for target in targets:
        layers = target_layers(params.target)
        last = add_layers(model, layers, branch_nodes, target)
        output = '%s_%s' % (target, 'y')
        model.add_output(input=last[0], name=output)
        outputs.append(output)

    if compile:
        optimizer = optimizer_from_params(params)
        loss = loss_from_ids(model.output_order)
        model.compile(loss=loss, optimizer=optimizer)

    return model


def loss_from_ids(ids):
    loss = dict()
    for x in ids:
        if x.startswith('s'):
            loss[x] = 'rmse'
        else:
            loss[x] = 'binary_crossentropy'
    return loss


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


def optimizer_from_json(path):
    with open(path, 'r') as f:
        config = f.read()
    return optimizer_from_config(config)


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
