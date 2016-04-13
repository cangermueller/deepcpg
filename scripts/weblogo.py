#!/usr/bin/env python

import argparse
import sys
import logging
import os
import os.path as pt
import pandas as pd


weblogo_opts = '-X NO -Y NO --errorbars NO --fineprint ""'
weblogo_opts += ' -C "#CB2026" A A'
weblogo_opts += ' -C "#34459C" C C'
weblogo_opts += ' -C "#FBB116" G G'
weblogo_opts += ' -C "#0C8040" T T'


def label_filt(path, tomtom=None, nb_top=0):
    filt = int(pt.basename(path).split('_')[0].replace('filter', ''))
    #  label = '%02d' % (filt)
    label = ''
    if tomtom is None:
        return label
    d = tomtom.loc[tomtom.filt == filt]
    if len(d) == 0:
        return label
    d = d.sort_values('q.value')
    label += '%s' % (d.iloc[0]['name'])
    if len(d) > 1 and nb_top > 0:
        label += ' (%s)' % (', '.join(d.iloc[1:(nb_top +1)]['name']))
    return label


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
            description='Create motif plots from FASTA files with weblogo')
        p.add_argument(
            'fasta_file',
            help='Fasta files',
            nargs='+')
        p.add_argument(
            '-o', '--out_dir',
            help='Output directory')
        p.add_argument(
            '-s', '--suffix',
            help='Suffix appended to file base')
        p.add_argument(
            '--tomtom_file',
            help='Label motif by TomTom file')
        p.add_argument(
            '--options',
            help='Weblogo options',
            default=weblogo_opts)
        p.add_argument(
            '--format',
            help='Output format',
            default='pdf')
        p.add_argument(
            '--args',
            help='Further arguments passed to weblogo')
        p.add_argument(
            '--test',
            help='Print command without executing it',
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

        tomtom = None
        if opts.tomtom_file is not None:
            tomtom = pd.read_table(opts.tomtom_file)

        for fasta_file in opts.fasta_file:
            out_file = pt.splitext(fasta_file)[0]
            if opts.suffix is not None:
                out_file += opts.suffix
            out_file = '%s.%s' % (out_file, opts.format)
            if opts.out_dir is not None:
                out_file = pt.join(opts.out_dir, pt.basename(out_file))
            cmd = 'weblogo {opts} -s large < {inp} > {out}'
            cmd += ' -F {f} {an} 2> /dev/null'
            if tomtom is None:
                an = ''
            else:
                an = '--label "%s"' % (label_filt(fasta_file, tomtom, nb_top=0))
            cmd = cmd.format(opts=opts.options,
                             inp=fasta_file,
                             out=out_file,
                             an=an, f=opts.format)
            if opts.args is not None:
                cmd += ' %s' % (opts.args)
            print(cmd)
            if not opts.test:
                os.system(cmd)

        return 0

if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
