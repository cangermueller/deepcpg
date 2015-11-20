import h5py as h5
import numpy as np
from predict.models.dnn_mc_cpg_seq.utils import DataReader

class TestDataReader(object):

    def make_file(self, path, chromos):
        f = h5.File(path, 'w')
        for chromo, l in chromos.items():
            f['/%s/pos' % (chromo)] = np.empty(l, dtype='bool')
        f.close()

    def test_reader(self):
        chromos = {'1': 10, '2': 5, '3': 3}
        path = 'data_reader.h5'
        self.make_file(path, chromos)

        print()
        r = DataReader(path, chunk_size=3, shuffle=True, loop=True)
        for chromo, i, j in r:
            print(chromo, i, j)
