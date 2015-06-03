import pdb
import sys
import os
import os.path as pt
import pandas as pd

sys.path.insert(0, pt.join(pt.dirname(pt.realpath(__file__)), '../predict'))
import cpgs as cp

class TestCpgs(object):

    def __chdir(self, dir):
        dir = pt.join('data/test_cpgs', dir)
        os.chdir(dir)

    def __unchdir(self):
        os.chdir('../../../')

    def __run(self, args):
        args = args.split()
        args.insert(0, 'cpgs.py')
        return cp.Cpgs().run(args)

    def test_cpgs(self):
        self.__chdir('test_cpgs')
        args = 'seq.h5 -o cpgs.txt'
        assert self.__run(args) == 0
        t1 = pd.read_table('cpgs.txt', header=False)
        t2 = pd.read_table('cpgs0.txt', header=False)
        assert (t1 == t2).all().all()
        os.remove('cpgs.txt')
        self.__unchdir()
