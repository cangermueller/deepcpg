#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import numpy as np
import h5py as h5
import theano as th

import predict.models.dnn.utils as ut
import predict.models.dnn.model as mod


def linear_weights(wlen, start=0.1):
    w = np.linspace(start, 1, np.ceil(wlen / 2))
    w = np.hstack((w, w[:-1][::-1]))
    return (w)


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
            description='Compute filter activations')
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
            '--conv_node',
            help='Convolutional node',
            default='s_c1')
        p.add_argument(
            '--outputs',
            help='What to store in output file',
            choices=['x', 'z', 'act'],
            default=['x', 'z', 'act'])
        p.add_argument(
            '--fun',
            help='Apply function to activations',
            choices=['mean', 'wmean', 'max'])
        p.add_argument(
            '--occ_input',
            help='Input to be occluded')
        p.add_argument(
            '--occ_nb',
            help='Number of occlusions',
            type=int,
            default=1)
        p.add_argument(
            '--occ_step',
            help='Occlusion step size',
            type=int)
        p.add_argument(
            '--occ_wlen',
            help='Occlusion window length',
            type=int,
            default=20)
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
            help='Maximum # samples',
            type=int)
        p.add_argument(
            '--batch_size',
            help='Batch size',
            type=int,
            default=128)
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

        def read_data(path):
            f = ut.open_hdf(path, cache_size=opts.max_mem)
            data = dict()
            for k, v in f['data'].items():
                data[k] = v
            for k, v in f['pos'].items():
                data[k] = v
            return (f, data)

        targets = ut.read_targets(opts.data_file)
        data_file, data = read_data(opts.data_file)

        if opts.chromo is not None:
            sel = data['chromo'].value == str(opts.chromo).encode()
            if opts.start is not None:
                sel &= data['pos'].value >= opts.start
            if opts.end is not None:
                sel &= data['pos'].value <= opts.end
            if sel.sum() == 0:
                log.warn('No samples satisfy filter!')
                return 0
            log.info('Select %d samples' % (sel.sum()))
            for k in data.keys():
                if len(data[k].shape) > 1:
                    data[k] = data[k][sel, :]
                else:
                    data[k] = data[k][sel]

        def to_view(d):
            for k in d.keys():
                d[k] = ut.ArrayView(d[k], stop=opts.nb_sample)

        to_view(data)
        log.info('%d samples' % (list(data.values())[0].shape[0]))

        log.info('Load model')
        model = mod.model_from_list(opts.model)
        conv_node = model.nodes[opts.conv_node]

        log.info('Store filter weights')
        out_file = h5.File(opts.out_file, 'w')
        out_group = out_file
        g = out_group.create_group('filter')
        g['weights'] = conv_node.get_weights()[0]
        g['bias'] = conv_node.get_weights()[1]

        log.info('Compute activations')
        x = [model.get_input(train=False)[x] for x in model.input_order]
        f_act = th.function(x, conv_node.get_output(train=False))
        f_z = model._predict
        ins = data
        nb_sample = ins[model.input_order[0]].shape[0]
        targets = ut.target_id2name(model.output_order, targets)
        out_group['targets'] = [x.encode() for x in targets]

        def write_hdf(x, path, idx, dtype=None, compression=None):
            if path not in out_group:
                if dtype is None:
                    dtype = x.dtype
                out_group.create_dataset(
                    name=path,
                    shape=[nb_sample] + list(x.shape[1:]),
                    dtype=dtype,
                    compression=compression
                )
            out_group[path][idx] = x

        if opts.fun is not None and opts.fun == 'wmean':
            win_weights = linear_weights(data['s_x'].shape[1])
        nb_batch = int(np.ceil(nb_sample / opts.batch_size))
        c = max(1, int(np.ceil(nb_batch / 50)))
        batch = 0
        for batch_start in range(0, nb_sample, opts.batch_size):
            batch += 1
            if batch == 1 or batch == nb_batch or batch % c == 0:
                print('%5d / %d (%.1f%%)' % (batch, nb_batch,
                                             batch / nb_batch * 100))

            batch_idx = slice(batch_start,
                              min(nb_sample, batch_start + opts.batch_size))
            ins_batch = {x: ins[x][batch_idx] for x in ins.keys()}
            ins_f = [ins_batch[x] for x in model.input_order]
            if len(ins_f) == 1:
                ins_f = ins_f[0]

            if 'z' in opts.outputs:
                out_z = f_z(ins_f)
                out_z = np.hstack(out_z)
                write_hdf(out_z, 'z', batch_idx)

            out_act = f_act(ins_f)
            if opts.fun is not None:
                if opts.fun == 'mean':
                    out_act = out_act.mean(axis=1)
                elif opts.fun == 'wmean':
                    out_act = np.average(out_act, axis=1, weights=win_weights)
                else:
                    out_act = out_act.max(axis=1)
            write_hdf(out_act, 'act', batch_idx, 'float16')

            for k in ['pos', 'chromo']:
                write_hdf(ins_batch[k], k, batch_idx)
            if 'x' in opts.outputs:
                for k in model.input_order:
                    x = ins_batch[k]
                    if k == 's_x':
                        # one-hot to int to reduce storage
                        x = x.argmax(axis=2).astype('int8')
                    write_hdf(x, k, batch_idx)

            if opts.occ_input is not None:
                X = ins_batch[opts.occ_input]
                seq_len = X.shape[1]
                seq_dim = X.shape[2]
                occ_step = seq_len // opts.occ_nb
                occ_idx = list(range(0, seq_len, occ_step))[:opts.occ_nb]
                center = seq_len // 2
                if center not in occ_idx:
                    occ_idx.append(center)
                    if len(occ_idx) > opts.occ_nb:
                        occ_idx = occ_idx[1:]

                batch_size = batch_idx.stop - batch_idx.start
                occ_z = np.empty((batch_size, len(occ_idx), out_z.shape[1]),
                                 dtype='float16')
                occ_act = np.empty((batch_size, len(occ_idx), out_act.shape[2]),
                                   dtype='float16')
                occ_del = opts.occ_wlen // 2

                for iidx, idx in enumerate(occ_idx):
                    X_occ = np.array(X, dtype='float16')
                    s = slice(max(0, idx - occ_del),
                              min(seq_len, idx + occ_del))
                    # Mutate sequence window
                    X_occ[:, s].fill(1 / seq_dim)
                    y = f_z(X_occ)
                    y = np.hstack(y)
                    occ_z[:, iidx] = y - out_z
                    y = f_act(X_occ)
                    occ_act[:, iidx] = y[:, idx] - out_act[:, idx]

                write_hdf(occ_z, '/occ/z', batch_idx)
                write_hdf(occ_act, '/occ/act', batch_idx, 'float16')
                if '/occ/idx' not in out_group:
                    out_group['/occ/idx'] = occ_idx

        data_file.close()
        out_file.close()
        log.info('Done!')

        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
