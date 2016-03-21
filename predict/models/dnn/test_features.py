#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import pandas as pd
import numpy as np
import keras.models as km
import theano as th

import predict.evaluation as pe
import predict.models.dnn.utils as ut
import predict.models.dnn.model as mod


def format_progress(batch_index, nb_batch, s=20):
    s = max(1, int(np.ceil(nb_batch / s)))
    f = None
    if batch_index == 1 or batch_index == nb_batch or batch_index % s == 0:
        f = '%5d / %d (%.1f%%)' % (batch_index, nb_batch,
                                   batch_index / nb_batch * 100)
    return f


def predict(model, data, batch_size=128, callbacks=[], log=print, f=None,
            output_order=None, cpg_nodist=False, cpg_wlen=None, seq_wlen=None,
            cpg_target=None):
    if f is None:
        f = model._predict
    if output_order is None:
        output_order = model.output_order
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
        batch_ids = index_array[batch_start:batch_end]
        batch_ids = batch_ids.copy()
        batch_ids.sort()
        batch_ids = list(batch_ids)
        ins_batch = km.slice_X(ins, batch_ids)
        if cpg_nodist:
            x = ins_batch[model.input_order.index('c_x')]
            x[:, 1].fill(0)
        if cpg_wlen is not None:
            x = ins_batch[model.input_order.index('c_x')]
            ctr = x.shape[3] // 2
            wdel = cpg_wlen // 2
            x[:, 0, :, :(ctr - wdel)] = 0.5
            x[:, 0, :, (ctr + wdel + 1):] = 0.5
            x[:, 1, :, :(ctr - wdel)] = 0
            x[:, 1, :, (ctr + wdel + 1):] = 0
        if cpg_target is not None:
            x = ins_batch[model.input_order.index('c_x')]
            i = [x for x in range(x.shape[2]) if x != cpg_target]
            x[:, 0, i] = 0.5
            x[:, 1, i] = 0
        if seq_wlen is not None:
            x = ins_batch[model.input_order.index('s_x')]
            ctr = x.shape[1] // 2
            wdel = seq_wlen // 2
            x[:, :(ctr - wdel)] = 0.25
            x[:, (ctr + wdel + 1):] = 0.25



        batch_outs = f(*ins_batch)
        if type(batch_outs) != list:
            batch_outs = [batch_outs]
        if batch_index == 0:
            for batch_out in batch_outs:
                shape = (nb_sample,) + batch_out.shape[1:]
                outs.append(np.zeros(shape))

        for i, batch_out in enumerate(batch_outs):
            outs[i][batch_start:batch_end] = batch_out

    return dict(zip(output_order, outs))


def evaluate(y, z, targets, *args, **kwargs):
    p = pe.evaluate_all(y, z, *args, **kwargs)
    p.index = ut.target_id2name(p.index.values, targets)
    p.index.name = 'target'
    p.reset_index(inplace=True)
    p.sort_values('target', inplace=True)
    return p


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
            description='Make prediction on data set')
        p.add_argument(
            'data_file',
            help='Data file')
        p.add_argument(
            '--model',
            help='Model files',
            nargs='+')
        p.add_argument(
            '-o', '--out_file',
            help='Output file')
        p.add_argument(
            '--cpg_nodist',
            help='No CpG distance',
            action='store_true')
        p.add_argument(
            '--cpg_wlen',
            help='CpG window length',
            type=int)
        p.add_argument(
            '--seq_wlen',
            help='Sequence window length',
            type=int)
        p.add_argument(
            '--target',
            help='Test target without taking other targets into account')
        p.add_argument(
            '--batch_size',
            help='Batch size',
            type=int,
            default=128)
        p.add_argument(
            '--chromo',
            help='Chromosome')
        p.add_argument(
            '--start',
            help='Start position',
            type=int)
        p.add_argument(
            '--end',
            help='End position',
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
        pd.set_option('display.width', 150)

        log.info('Load model')
        model = mod.model_from_list(opts.model)

        log.info('Load data')
        targets = ut.read_targets(opts.data_file)
        data_file, data = ut.read_hdf(opts.data_file, opts.max_mem)
        if opts.chromo is not None:
            log.info('Select data')
            ut.select_data(data, opts.chromo, opts.start, opts.end)
            log.info('%d sites selected' % (len(data['pos'])))
        ut.to_view(data, stop=opts.nb_sample)

        def to_view(d):
            for k in d.keys():
                d[k] = ut.ArrayView(d[k], stop=opts.nb_sample)

        to_view(data)
        log.info('%d samples' % (list(data.values())[0].shape[0]))

        if opts.target is None:
            fun = model._predict
            output_order = model.output_order
            target_idx = None
        else:
            target_idx = targets['name'].index(opts.target)
            assert target_idx >= 0
            target_id = targets['id'][target_idx] + '_y'
            output_order = [target_id]
            log.info('Compile function for target %s (%s)' % (opts.target,
                                                              target_id))
            y = model.outputs[target_id].get_output(True)
            x = [model.inputs[name].input for name in model.input_order]
            fun = th.function(inputs=x, outputs=y, allow_input_downcast=True)

        log.info('Predict')
        z = predict(model, data, batch_size=opts.batch_size,
                    cpg_nodist=opts.cpg_nodist, cpg_wlen=opts.cpg_wlen,
                    seq_wlen=opts.seq_wlen, f=fun, output_order=output_order,
                    cpg_target=target_idx)

        e = evaluate(data, z, targets)
        print('Performance:')
        print(e.to_string(index=False))

        log.info('Write')
        ut.write_z(data, z, targets, opts.out_file)

        data_file.close()
        log.info('Done!')

        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
