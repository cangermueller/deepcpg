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
            description='Run script in directories')
        p.add_argument(
            'script_file',
            help='Script to be executed')
        p.add_argument(
            'target_dirs',
            help='Target directories',
            nargs='+')
        p.add_argument(
            '-d', '--sub_dir',
            help='Execute in subdirectory')
        p.add_argument(
            '-n', '--script_name',
            help='Name of script in target directory')
        p.add_argument(
            '--rm',
            help='Remove target directory',
            action='store_true')
        p.add_argument(
            '-r', '--run',
            help='Run model',
            choices=['none', 'local', 'cpu', 'gpu'],
            default='local')
        p.add_argument(
            '-A', '--account',
            help='SLURM account',
            choices=['SL2', 'SL3', 'SL4'],
            default='SL3')
        p.add_argument(
            '-t', '--time',
            help='Maximum run time',
            type=int,
            default=2)
        p.add_argument(
            '-m', '--memory',
            help='Maximum memory',
            type=int,
            default=16000)
        p.add_argument(
            '--args',
            help='Additional SLURM arguments',
            nargs='+')
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

        script_name = opts.script_name
        if script_name is None:
            script_name = pt.basename(opts.script_file)

        for target_dir in opts.target_dirs:
            job_name = pt.basename(target_dir)
            if opts.sub_dir:
                target_dir = pt.join(target_dir, opts.sub_dir)
            run_file = pt.join(target_dir, script_name)
            log.info(run_file)
            if opts.rm:
                shutil.rmtree(target_dir)
            os.makedirs(target_dir, exist_ok=True)
            shutil.copyfile(opts.script_file, run_file)
            os.system('chmod 744 %s' % (run_file))
            run_file_local = pt.basename(run_file)

            cmd = './%s' % (run_file_local)
            if pt.splitext(run_file_local)[1] == '.Rmd':
                cmd = 'rmd.R %s' % (cmd)

            if opts.run == 'none':
                continue
            elif opts.run == 'local':
                cmd = cmd
            else:
                scmd = 'sbatch --mem={mem} -J {job} -o {log}.out -e {log}.err' +\
                    ' --time={time}:00:00 -A {acc} {args} {sfile} {cmd}'
                if opts.run == 'cpu':
                    account = 'STEGLE-%s' % (opts.account)
                    sfile = os.getenv('scpu')
                else:
                    account = 'STEGLE-%s-GPU' % (opts.account)
                    sfile = os.getenv('sgpu')
                if opts.args is None:
                    args = ''
                else:
                    args = ' '.join(opts.args)
                cmd = scmd.format(mem=opts.memory, job=job_name,
                                 log=run_file_local, time=opts.time,
                                 acc=account, sfile=sfile, args=args,
                                 cmd=cmd)
            print(cmd)
            if not opts.test:
                h = os.getcwd()
                os.chdir(target_dir)
                os.system(cmd)
                os.chdir(h)

        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
