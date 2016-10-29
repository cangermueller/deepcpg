from keras import layers as kl
from keras import models as km
from keras import regularizers as kr

from . import dna as mdna
from . import cpg as mcpg
from .utils import Model

from ..utils import get_from_module, as_list


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


class Joint02(JointModel):

    def __init__(self, mode='concat', nb_hidden=1024, *args, **kwargs):
        super(JointModel, self).__init__(*args, **kwargs)
        self.mode = mode
        self.nb_hidden = nb_hidden

    def __call__(self, models):
        model = km.Sequential()
        model.add(kl.Merge(models, mode=self.mode))
        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        model.add(kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg))
        model.add(kl.BatchNormalization())
        model.add(kl.Activation('relu'))
        return model

        #  outputs = []
        #  for model in models:
        #      outputs.extend(as_list(model.input))
        #  x = kl.Merge(models, mode=self.mode)(inputs)


        #  w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        #  x = kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg)(x)
        #  x = kl.BatchNormalization()(x)
        #  x = kl.Activation('relu')(x)
        #  x = kl.Dropout(self.dropout)(x)

        #  return x


def get(name):
    return get_from_module(name, globals())
