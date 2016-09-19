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
            description='Creates models')
        p.add_argument(
            'config_files',
            help='Model configuration files',
            nargs='+')
        p.add_argument(
            '-t', '--temp_file',
            help='Training template file',
            default='./temp/test.sh')
        p.add_argument(
            '-o', '--out_dir',
            help='Output director',
            default='./models')
        p.add_argument(
            '-p', '--prefix',
            help='Model prefix')
        p.add_argument(
            '-r', '--run',
            help='Run model',
            choices=['none', 'local', 'cpu', 'gpu'],
            default='local')
        p.add_argument(
            '-a', '--account',
            help='SLURM account',
            choices=['SL2', 'SL3'],
            default='SL3')
        p.add_argument(
            '--time',
            help='Maximum run time',
            type=int,
            default=12)
        p.add_argument(
            '--test',
            help='Print command without executing',
            action='store_true')
        p.add_argument(
            '-R', '--no_remove',
            help='Do not remove directory if existing',
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

        for config_file in opts.config_files:
            name = pt.splitext(pt.basename(config_file))[0]
            prefix = opts.prefix
            if prefix is None:
                prefix = pt.splitext(pt.basename(opts.temp_file))[0]
            name = '%s-%s' % (prefix, name)
            out_dir = pt.join(opts.out_dir, name, 'train')
            log.info(out_dir)

            if not opts.no_remove and pt.exists(out_dir):
                shutil.rmtree(out_dir)
            os.makedirs(out_dir, exist_ok=True)
            run_file = pt.join(out_dir, 'train.sh')
            shutil.copyfile(opts.temp_file, run_file)
            os.system('chmod 744 %s' % (run_file))
            shutil.copyfile(config_file, pt.join(out_dir, 'configs.yaml'))
            run_file = pt.basename(run_file)
            if opts.run == 'none':
                continue
            elif opts.run == 'local':
                cmd='bash %s' % (run_file)
            else:
                cmd = 'sbatch --mem=32000 -J {job} -o {log}.out -e {log}.err' +\
                    ' --time={time}:00:00 -A {acc} {sfile} ./{rfile}'
                if opts.run == 'cpu':
                    account = 'STEGLE-SL3'
                    sfile = os.getenv('scpu')
                else:
                    account = 'STEGLE-%s-GPU' % (opts.account)
                    sfile = os.getenv('sgpu')
                cmd = cmd.format(job=name, log=run_file, time=opts.time,
                                    acc=account, sfile=sfile, rfile=run_file)

            print(cmd)
            if not opts.test:
                h = os.getcwd()
                os.chdir(out_dir)
                os.system(cmd)
                os.chdir(h)


        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
