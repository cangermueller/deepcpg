#!/usr/bin/env python

import argparse
import sys
import logging
import os
import os.path as pt
import shutil


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
            description='Run rmd script')
        p.add_argument(
            'rmd_file',
            help='RMD file')
        p.add_argument(
            '--copy',
            help='Copy RMD file to target file')
        p.add_argument(
            '-d', '--depends',
            help='Dependencies to be copied',
            nargs='+')
        p.add_argument(
            '-f', '--format',
            help='Output format',
            default='html',
            choices=['html', 'pdf', 'word'])
        p.add_argument(
            '--cmd',
            help='R command')
        p.add_argument(
            '--test',
            help='Print command without executing',
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

        rmd_file = opts.rmd_file

        copy = []
        if opts.copy is not None:
            copy.append([rmd_file, opts.copy])
            rmd_file = opts.copy
        out_dir = pt.dirname(rmd_file)
        if opts.depends is not None:
            for src in opts.depends:
                dst = pt.join(out_dir, pt.basename(src))
                copy.append([src, dst])
        for src, dst in copy:
            shutil.copyfile(src, dst)

        _format = opts.format
        out_file = '%s.%s' % (pt.splitext(pt.basename(rmd_file))[0],
                              opts.format)
        Rcmd = ''
        if opts.cmd is not None:
            Rcmd = '%s;' % (opts.cmd)
        cmd = "library(rmarkdown); {c} render('{r}', output_file='{o}', output_format='{f}_document')"
        cmd = cmd.format(c=Rcmd, r=rmd_file, o=out_file, f=_format)
        cmd = 'Rscript -e "%s"' % (cmd)
        print(cmd)
        if not opts.test:
            os.system(cmd)

        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
