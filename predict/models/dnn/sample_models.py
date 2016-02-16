#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import yaml
import numpy as np
import scipy.stats as sps

from predict.models.dnn.params import ParamSampler


def eval_atom(v):
    if isinstance(v, str) and v.startswith('sps.'):
        v = eval(v)
    return v


def eval_dict(d):
    if isinstance(d, list):
        for i, v in enumerate(d):
            if isinstance(v, list) or isinstance(v, dict):
                eval_dict(v)
            else:
                d[i] = eval_atom(v)
    elif isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, list) or isinstance(v, dict):
                eval_dict(v)
            else:
                d[k] = eval_atom(v)


class App(object):

    def run(self, args):
        name = pt.basename(args[0])
        parser = self.create_parser(name)
        opts = parser.parse_args(args[1:])
        return self.main(name, opts)

    def create_parser(self, name):
        p = argparse.ArgumentParser(
            prog=name,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Sample model parameter files')
        p.add_argument(
            'temp_file',
            help='Model template YAML file')
        p.add_argument(
            '-o', '--out_base',
            help='Output basename',
            default='./m')
        p.add_argument(
            '-n', '--nb_sample',
            help='Number of samples',
            type=int,
            default=1)
        p.add_argument(
            '--offset',
            help='Offset of model number',
            type=int,
            default=0)
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

        np.random.seed(opts.seed)

        with open(opts.temp_file) as f:
            temp = yaml.load(f.read())
        eval_dict(temp)
        import ipdb; ipdb.set_trace()

        for i, param in enumerate(ParamSampler(temp, opts.nb_sample)):
            t = '%s%03d.yaml' % (opts.out_base, opts.offset + i)
            log.info(t)
            param.to_yaml(t)

        return 0

if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
