#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import numpy as np
import theano as th
import h5py as h5
import theano.tensor as T
import keras.models as km

import predict.io as io
import predict.models.dnn.utils as ut
import predict.models.dnn.model as mod


def progress(batch_index, nb_batch, s=20):
    s = max(1, int(np.ceil(nb_batch / s)))
    f = None
    batch_index += 1
    if batch_index == 1 or batch_index == nb_batch or batch_index % s == 0:
        f = '%5d / %d (%.1f%%)' % (batch_index, nb_batch,
                                   batch_index / nb_batch * 100)
    return f


def write_loop(ins, fun, write_fun, batch_size=128, callbacks=[], log=print):
    nb_sample = len(ins[0])
    batches = km.make_batches(nb_sample, batch_size)
    index_array = np.arange(nb_sample)
    nb_batch = len(batches)
    for batch_index, (batch_start, batch_end) in enumerate(batches):
        if log is not None:
            s = progress(batch_index, nb_batch)
            if s is not None:
                log(s)
        for callback in callbacks:
            callback(batch_index, len(batches))
        batch_ids = list(index_array[batch_start:batch_end])
        ins_batch = km.slice_X(ins, batch_ids)

        batch_outs = fun(*ins_batch)
        write_fun(batch_outs, batch_start, batch_end)


def agg_score(s, wlen=None, fun=None):
    if wlen is not None:
        s = s[:, io.slice_center(s.shape[1], wlen)]
    if fun is not None:
        if fun == 'max':
            f = np.max
        else:
            f = np.mean
        s = f(s, axis=1)
    return s


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
            description='Effect of nodes')
        p.add_argument(
            'data_file',
            help='Data file')
        p.add_argument(
            '--model',
            help='Model file',
            nargs='+')
        p.add_argument(
            '-o', '--out_file',
            help='Output file',
            default='snp.h5')
        p.add_argument(
            '-x', '--effect_node',
            help='Effect node',
            default='s_x')
        p.add_argument(
            '--outputs',
            help='Output options',
            choices=['x'],
            nargs='+',
            default=[])
        p.add_argument(
            '--effects',
            help='Effects',
            choices=['all', 'mean', 'var', 'disp'],
            default=['mean'],
            nargs='+')
        p.add_argument(
            '--wlen',
            help='Slicing window length',
            type=int)
        p.add_argument(
            '--char',
            help='Per character effect',
            action='store_true')
        p.add_argument(
            '--agg_fun',
            help='Aggregation function',
            choices=['mean', 'max'])
        p.add_argument(
            '--target',
            help='Target basename',
            default='ser_w3000')
        p.add_argument(
            '--chromo',
            help='Chromosome')
        p.add_argument(
            '--start',
            help='Start position',
            type=int)
        p.add_argument(
            '--stop',
            help='Stop position',
            type=int)
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

        log.info('Load data')
        data_file, data = ut.read_data(opts.data_file, opts.max_mem)
        sel = io.select_region(data['chromo'], data['pos'],
                               chromo=opts.chromo, start=opts.start,
                               stop=opts.stop, nb_sample=opts.nb_sample)
        ut.to_view(data, start=sel.start, stop=sel.stop)

        ut.to_view(data, stop=opts.nb_sample)
        nb_sample = list(data.values())[0].shape[0]
        log.info('%d samples' % (nb_sample))

        log.info('Load model')
        model = mod.model_from_list(opts.model)

        out_file = h5.File(opts.out_file, 'w')

        tar = ut.read_targets(opts.data_file)
        tar_n2i = {n: i for i, n in zip(tar['id'], tar['name'])}
        tar_mean = '%s_mean' % (opts.target)
        tar_var = '%s_var' % (opts.target)

        ins = model.get_input(train=False)
        input_nodes = [ins[x] for x in model.input_order]
        effect_node = ins[opts.effect_node]
        output_nodes = model.get_output(train=False)

        log.info('Compile')
        eff_list = list()
        gs = []
        for effect in opts.effects:
            if effect == 'mean':
                y = output_nodes[tar_n2i[tar_mean] + '_y']
            elif effect == 'var':
                y = output_nodes[tar_n2i[tar_var] + '_y']
            elif effect == 'disp':
                m = output_nodes[tar_n2i[tar_mean] + '_y']
                v = output_nodes[tar_n2i[tar_var] + '_y']
                y = T.sqr(m * (1 - m) - v)
            else:
                _ = list(output_nodes.values())
                y = _[0]
                for yy in _[1:]:
                    y += yy
                y = y / len(_)
            g = T.grad(T.mean(y), effect_node)
            if not opts.char:
                g = T.max(T.abs_(g), axis=2)
            gs.append(g)
            eff_list.append(effect)
        f = th.function(input_nodes, gs)

        def write_fun(x, batch_start, batch_end):
            for i, xi in enumerate(x):
                if opts.wlen is not None:
                    xi = xi[:, io.slice_center(xi.shape[1], opts.wlen)]
                if not (opts.char or opts.agg_fun is None):
                    _ = np.mean
                    if opts.agg_fun == 'max':
                        _ = np.max
                    xi = _(xi, axis=1)
                g = eff_list[i]
                if g in out_file:
                    g = out_file[g]
                else:
                    _ = [nb_sample] + list(xi.shape[1:])
                    g = out_file.create_dataset(g, data=np.zeros(_))
                g[batch_start:batch_end] = xi

        log.info('Compute ...')
        ins = [data[x] for x in model.input_order]
        write_loop(ins, f, write_fun, 1)

        out_file['chromo'] = data['chromo'][:]
        out_file['pos'] = data['pos'][:]
        if 'x' in opts.outputs:
            for x in model.input_order:
                d = data[x]
                if x == 's_x':
                    d = d.argmax(axis=2)
                out_file[x] = d

        out_file.close()
        data_file.close()
        log.info('Done!')
        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
