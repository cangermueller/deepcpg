from keras import layers as kl

from . import dna as mdna
from . import cpg as mcpg
from .utils import Model


"""TODO
Constant for init?
"""


class JointModel(Model):

    def reader(self, data_files, cpg_names, *args, **kwargs):
        super_reader = super(JointModel, self).reader
        for data in super_reader(data_files,
                                 use_dna=True,
                                 cpg_names=cpg_names,
                                 *args, **kwargs):
            yield data

    def inputs(self, dna_wlen, nb_context, cpg_wlen):
        inputs = []
        inputs.append(kl.Input(shape=(dna_wlen, 4),
                               name='dna'))
        inputs.append(kl.Input(shape=(nb_context, cpg_wlen),
                               name='cpg_context/state'))
        inputs.append(kl.Input(shape=(nb_context, cpg_wlen),
                               name='cpg_context/dist'))
        return inputs


class Joint01(JointModel):

    def __init__(self, *args, **kwargs):
        self.dna_model = mdna.Dna01(*args, **kwargs)
        self.cpg_model = mcpg.Cpg01(*args, **kwargs)

    def __call__(self, inputs):
        dna_stem = self.dna_model(inputs[:1])
        cpg_stem = self.cpg_model(inputs[1:])

        joint_stem = kl.merge([dna_stem, cpg_stem], mode='concat',
                              concat_axis=1)
        return joint_stem
