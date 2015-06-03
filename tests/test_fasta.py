import sys
import os.path as pt
import os
import pickle
import pdb
import h5py

sys.path.insert(0, pt.join(pt.dirname(pt.realpath(__file__)), '../predict'))
import cpgs as cp
import fasta
from fasta import FastaSeq


class TestFasta(object):

    def setup(self):
        self.data_dir = 'data/test_fasta'

    def test_parse_lines(self):
        s = []
        s.append('>seq1')
        s.append('l')
        s.append('line2')
        seqs = fasta.parse_lines(s)
        assert len(seqs) == 1
        assert seqs[0].head == '>seq1'
        assert seqs[0].seq == 'lline2'

        s = []
        s.append('>seq1')
        s.append('line1')
        s.append(' line2  ')
        s.append('')
        s.append('line3')
        s.append('')
        seqs = fasta.parse_lines(s)
        assert len(seqs) == 1
        assert seqs[0].head == '>seq1'
        assert seqs[0].seq == 'line1line2line3'

        s = []
        s.append('>seq1')
        s.append('line1')
        s.append('line2')
        s.append('')
        s.append('>seq2')
        s.append('')
        s.append('  line21 ')
        s.append('>seq3')
        s.append('>seq4')
        seqs = fasta.parse_lines(s)
        assert len(seqs) == 4
        assert seqs[0].head == '>seq1'
        assert seqs[0].seq == 'line1line2'
        assert seqs[1].head == '>seq2'
        assert seqs[1].seq == 'line21'
        assert seqs[2].head == '>seq3'
        assert seqs[2].seq == ''
        assert seqs[3].head == '>seq4'
        assert seqs[3].seq == ''

    def test_read_file(self):
        self.setup()
        filename = pt.join(self.data_dir, 'test_read_file/seq.fa')
        seq = fasta.read_file(filename)
        assert len(seq) == 2
        assert seq[1].head == '>seq2'

        filename = pt.join(self.data_dir, 'test_read_file/seq.fa.gz')
        seq = fasta.read_file(filename)
        assert len(seq) == 2
        assert seq[1].head == '>seq2'

    def __call(self, args):
        cmd = '../predict/fasta.py ' + args
        print(cmd)
        return os.system(cmd)

    def test_pk1(self):
        self.setup()
        data_dir = pt.join(self.data_dir, 'test_pk1')
        in_file = pt.join(data_dir, 'seq.fa.gz')
        out_file = pt.join(data_dir, 'seq.pk1')
        args = '%s --out_file %s' % (in_file, out_file)
        assert self.__call(args) == 0
        seqs = pickle.load(open(out_file, 'rb'))
        assert len(seqs) == 2
        assert seqs[1] == 'Sequence2'
        os.remove(out_file)

    def test_h5(self):
        self.setup()
        data_dir = pt.join(self.data_dir, 'test_h5')

        in_file = pt.join(data_dir, 'seq.fa.gz')
        out_file = pt.join(data_dir, 'seq.h5')
        args = '%s --out_file %s' % (in_file, out_file)
        assert self.__call(args) == 0
        assert self.__call(args) == 0
        f = h5py.File(out_file)
        assert len(f.keys()) == 2
        assert '/0' in f
        assert '/1' in f
        seq = f['/1'].value
        assert seq == 'Sequence2'
        os.remove(out_file)

        in_file = pt.join(data_dir, 'seq2.fa')
        out_file = pt.join(data_dir, 'seq2.h5')
        args = '%s --out_file %s' % (in_file, out_file + ':/chromo1')
        assert self.__call(args) == 0
        assert self.__call(args) == 0
        f = h5py.File(out_file)
        assert len(f.keys()) == 1
        seq = f['/chromo1'].value
        assert seq == 'AGGTTA'
        os.remove(out_file)
