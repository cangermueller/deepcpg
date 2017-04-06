#!/usr/bin/env python

"""Download a pre-trained model from DeepCpG model zoo.

Downloads a pre-trained model from the DeepCpG model zoo by its identifier.
Model descriptions can be found on online.

Examples
--------
Show available models:

.. code:: bash

    dcpg_download --show

Download DNA model trained on serum cells from Smallwood et al:

.. code:: bash

    dcpg_download.py
        Smallwood2014_serum_dna
        -o ./model
"""

from __future__ import print_function
from __future__ import division

import os
import sys

import argparse
import logging

from deepcpg.utils import make_dir


DATA_HOST = 'http://www.ebi.ac.uk/~angermue/deepcpg/alias'
MODEL_ZOO = 'https://github.com/cangermueller/deepcpg/blob/master/' \
    'docs/models.md'

MODELS = {
    'Smallwood2014_2i_dna': '51b5b3df82e5431a37794640647baafd',
    'Smallwood2014_2i_cpg': 'f89b2e8344012d73e95504da06bcf378',
    'Smallwood2014_2i_joint': '7c8fbb955d620d994f391630ef0b909c',
    'Smallwood2014_serum_dna': '1754b5bbc21a8257663acc52e657f69c',
    'Smallwood2014_serum_cpg': '33bc504f24df2e7a0380ef75aaa70e59',
    'Smallwood2014_serum_joint': 'e4b82088e980cb26b87a157a4f69abc0',
    'Hou2016_HCC_dna': '260e4c19cef65fd36f7e7e3d7edd2c15',
    'Hou2016_HCC_cpg': '78730fac7c4bd2c3a43b7cf9ef99f547',
    'Hou2016_HCC_joint': 'f58d95caa1c6dfb4d5d796af855292ae',
    'Hou2016_HepG2_dna': 'e5ffbf52cef081c9bf58e09ba873fd95',
    'Hou2016_HepG2_cpg': '0e463dc20d82f50d8b1fe45e3c3c1cec',
    'Hou2016_HepG2_joint': 'e84cc69d34789f05e3a1a1c9f142c737',
    'Hou2016_mESC_dna': 'e08c0dc73cff5a6ff3c42cd6d359b609',
    'Hou2016_mESC_cpg': '961db2f0f32dff01eb7806a929be48f0',
    'Hou2016_mESC_joint': '585abb5dae563229f338002b70f44c7c'
}


def run(command):
    exit = os.system(command)
    if exit:
        raise RuntimeError('Command executing "%s"!' % command)


class App(object):

    def run(self, args):
        name = os.path.basename(args[0])
        parser = self.create_parser(name)
        opts = parser.parse_args(args[1:])
        return self.main(name, opts)

    def create_parser(self, name):
        p = argparse.ArgumentParser(
            prog=name,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Downloads a pre-trained model from DeepCpG model zoo.')
        p.add_argument(
            'model_id',
            help='Model identifier from DeepCpG model zoo',
            choices=sorted(list(MODELS.keys())),
            nargs='?')
        p.add_argument(
            '-o', '--out_dir',
            help='Output directory',
            default='./model')
        p.add_argument(
            '-s', '--show',
            help='Show name of available models',
            action='store_true')
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

        if opts.show:
            print('Available models: %s' % MODEL_ZOO)
            for name in sorted(list(MODELS.keys())):
                print(name)
            return 0

        if not opts.model_id:
            raise ValueError('Model ID required!')

        if opts.model_id not in MODELS:
            raise ValueError('Invalid model ID "%s"!' % opts.model_id)

        log.info('Downloading model ...')
        model_url = "%s/%s" % (DATA_HOST, MODELS[opts.model_id])
        log.info('Model URL: %s' % model_url)
        make_dir(opts.out_dir)
        zip_file = os.path.join(opts.out_dir, 'model.zip')
        run('wget "%s" -O %s' % (model_url, zip_file))
        run('unzip -o %s -d %s' % (zip_file, opts.out_dir))
        os.remove(zip_file)

        log.info('Done!')

        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
