#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import pandas as pd
import numpy as np
import keras.models as km
import theano as th
import h5py as h5

import predict.models.dnn.utils as ut
import predict.models.dnn.model as mod


def format_progress(batch_index, nb_batch, s=20):
    s = max(1, int(np.ceil(nb_batch / s)))
    f = None
    if batch_index == 1 or batch_index == nb_batch or batch_index % s == 0:
        f = '%5d / %d (%.1f%%)' % (batch_index, nb_batch,
                                   batch_index / nb_batch * 100)
    return f


def predict_loop(model, data, batch_size=128, callbacks=[], log=print, f=None):
    if f is None:
        f = model._predict
    ins = [data[name] for name in model.input_order]
    nb_sample = len(ins[0])
    outs = []
    batches = km.make_batches(nb_sample, batch_size)
    index_array = np.arange(nb_sample)
    nb_batch = len(batches)
    for batch_index, (batch_start, batch_end) in enumerate(batches):
        if log is not None:
            s = format_progress(batch_index, nb_batch)
            if s is not None:
                log(s)
        for callback in callbacks:
            callback(batch_index, len(batches))
        batch_ids = list(index_array[batch_start:batch_end])
        ins_batch = km.slice_X(ins, batch_ids)

        batch_outs = f(*ins_batch)
        if type(batch_outs) != list:
            batch_outs = [batch_outs]

        if batch_index == 0:
            for batch_out in batch_outs:
                shape = (nb_sample,) + batch_out.shape[1:]
                outs.append(np.zeros(shape))

        for i, batch_out in enumerate(batch_outs):
            outs[i][batch_start:batch_end] = batch_out

    return dict(zip(model.output_order, outs))


def write_loop(ins, fun, write_fun, batch_size=128, callbacks=[], log=print):
    nb_sample = len(ins[0])
    batches = km.make_batches(nb_sample, batch_size)
    index_array = np.arange(nb_sample)
    nb_batch = len(batches)
    for batch_index, (batch_start, batch_end) in enumerate(batches):
        if log is not None:
            s = format_progress(batch_index, nb_batch)
            if s is not None:
                log(s)
        for callback in callbacks:
            callback(batch_index, len(batches))
        batch_ids = list(index_array[batch_start:batch_end])
        ins_batch = km.slice_X(ins, batch_ids)

        batch_outs = fun(*ins_batch)
        write_fun(batch_outs, batch_start, batch_end)


def read_hdf(path, cache_size):
    f = ut.open_hdf(path, cache_size=cache_size)
    data = dict()
    for k, v in f['data'].items():
        data[k] = v
    for k, v in f['pos'].items():
        data[k] = v
    return (f, data)


def to_view(d, *args, **kwargs):
    for k in d.keys():
        d[k] = ut.ArrayView(d[k], *args, **kwargs)


class App(object):

    def run(self, args):
        name = pt.basename(args[0])
        parser = self.create_parser(name)
        opts = parser.parse_args(args[1:])
        self.opts = opts
        return self.main(name, opts)

    def create_parser(self, name):
        p = argparse.ArgumentParser(
            prog=name,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Compute node activations')
        p.add_argument(
            'data_file',
            help='Data file')
        p.add_argument(
            '--model',
            help='Model files',
            nargs='+')
        p.add_argument(
            '--node',
            help='Name of node')
        p.add_argument(
            '-o', '--out_file',
            help='Output file')
        p.add_argument(
            '--out_group',
            help='Output group')
        p.add_argument(
            '--tsne',
            help='Number of T-SNE components',
            type=int)
        p.add_argument(
            '--batch_size',
            help='Batch size',
            type=int,
            default=128)
        p.add_argument(
            '--nb_sample',
            help='Maximum # training samples',
            type=int)
        p.add_argument(
            '--max_mem',
            help='Maximum memory load',
            type=int,
            default=14000)
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

        if opts.node is None:
            raise 'Node required!'

        log.info('Load model')
        model = mod.model_from_list(opts.model)

        log.info('Compile activation function')
        ins = [model.get_input(train=False)[x] for x in model.input_order]
        node = model.nodes[opts.node]
        fun = th.function(ins, node.get_output(train=False))

        log.info('Load data')
        data_file, data = read_hdf(opts.data_file, opts.max_mem)
        to_view(data, stop=opts.nb_sample)
        nb_sample = data[model.input_order[0]].shape[0]
        log.info('%d samples' % (nb_sample))

        log.info('Initialize dataset')
        out_file = h5.File(opts.out_file, 'a')
        h = opts.out_group
        if h is None:
            h = opts.node
        if h in out_file:
            del out_file[h]
        out_group = out_file.create_group(h)
        for x in ['pos', 'chromo']:
            out_group[x] = data[x]
        out_shape = [nb_sample] + list(node.output_shape[1:])
        out_group.create_dataset('act', shape=out_shape, dtype='float32')

        def write_fun(x, batch_start, batch_end):
            out_group['act'][batch_start:batch_end] = x

        log.info('Compute activations')
        ins = [data[x] for x in model.input_order]
        write_loop(ins, fun, write_fun, batch_size=opts.batch_size)

        data_file.close()
        log.info('Done!')

        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
