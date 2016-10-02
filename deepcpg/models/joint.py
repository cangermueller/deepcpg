from keras import layers as kl

from . import dna as mdna
from . import cpg as mcpg
from .utils import Model


"""TODO
Constant for init?
"""


class JointModel(Model):

    def __init__(self, replicate_names, cpg_wlen=None, dna_wlen=None,
                 *args, **kwargs):
        super(JointModel, self).__init__(*args, **kwargs)
        self.replicate_names = replicate_names
        self.cpg_wlen = cpg_wlen
        self.dna_wlen = dna_wlen

    def reader(self, data_files, *args, **kwargs):
        super_reader = super(JointModel, self).reader
        for data in super_reader(data_files,
                                 use_dna=True,
                                 dna_wlen=self.dna_wlen,
                                 replicate_names=self.replicate_names,
                                 cpg_wlen=self.cpg_wlen,
                                 *args, **kwargs):
            yield data

    def inputs(self):
        inputs = []
        inputs.append(kl.Input(shape=(self.dna_wlen, 4),
                               name='dna'))
        shape = (len(self.replicate_names), self.cpg_wlen)
        inputs.append(kl.Input(shape=shape, name='cpg/state'))
        inputs.append(kl.Input(shape=shape, name='cpg/dist'))
        return inputs


class Joint01(JointModel):

    def __init__(self, replicate_names, cpg_wlen=None, dna_wlen=None,
                 *args, **kwargs):
        super(Joint01, self).__init__(replicate_names,
                                      cpg_wlen=cpg_wlen,
                                      dna_wlen=dna_wlen,
                                      *args, **kwargs)
        self.dna_model = mdna.Dna01(dna_wlen=dna_wlen, *args, **kwargs)
        self.cpg_model = mcpg.Cpg01(replicate_names=replicate_names,
                                    cpg_wlen=cpg_wlen,
                                    *args, **kwargs)

    def __call__(self, inputs):
        dna_stem = self.dna_model(inputs[:1])
        cpg_stem = self.cpg_model(inputs[1:])

        joint_stem = kl.merge([dna_stem, cpg_stem], mode='concat',
                              concat_axis=1)
        return joint_stem
