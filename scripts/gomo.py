#!/usr/bin/env python

import argparse
import sys
import logging
import os.path as pt
import re
import pandas as pd


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
            description='Parses GOMo html file')
        p.add_argument(
            'in_file',
            help='GOMo html file')
        p.add_argument(
            '-o', '--out_file',
            help='Output file csv')
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

        log.info('Parse file')
        tab = []
        in_file = open(opts.in_file, 'r')
        filt = None
        in_tab = False
        for line in in_file:
            if line.startswith('<h4>Overview</h4>'):
                in_tab = True
            if not in_tab:
                continue
            m = re.search(r'<a href="#motif_(\w+)">(\d+)</a>', line)
            if m:
                filt = m.group(1)
                nb_hit = m.group(2)
                continue
            if filt is None:
                continue
            if line.startswith('<span class='):
                for hit in line.split('<br>'):
                    _ = hit.split('\xa0')
                    if len(_) != 2:
                        break
                    go = _[0].split('>')[1]
                    go_text = _[1].split('>')[1]
                    tab.append([filt, nb_hit, go, go_text])
            if line.startswith('</table>'):
                in_tab = False
                break
        in_file.close()
        if len(tab) == 0:
            raise 'Invalid file format'
        tab = pd.DataFrame(tab, columns=['filt', 'nb_hit', 'go', 'go_term'])
        _ = tab.to_csv(opts.out_file, sep='\t', index=False)
        if _ is not None:
            print(_, end='')

        log.info('Done!')
        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
