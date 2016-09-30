from keras import layers as kl
from keras import regularizers as kr
from keras import models as km

from .utils import Model


class CpgModel(Model):

    def reader(self, data_files, cpg_names, *args, **kwargs):
        super_reader = super(CpgModel, self).reader
        for data in super_reader(data_files,
                                 use_dna=False,
                                 cpg_names=cpg_names,
                                 *args, **kwargs):
            yield data

    def inputs(self, nb_context, cpg_wlen):
        inputs = []
        inputs.append(kl.Input(shape=(nb_context, cpg_wlen),
                               name='cpg_context/state'))
        inputs.append(kl.Input(shape=(nb_context, cpg_wlen),
                               name='cpg_context/dist'))
        return inputs


class Cpg01(CpgModel):

    def _cpg_model(self, input):
        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(512, init='he_uniform', W_regularizer=w_reg)(input)
        x = kl.BatchNormalization(mode=2, axis=1)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)
        return km.Model(input=input, output=x)

    def __call__(self, inputs):
        x = kl.merge(inputs, mode='concat', concat_axis=2)

        cpg_input = kl.Input(shape=(int(x.get_shape()[-1]),))
        cpg_model = self._cpg_model(cpg_input)

        x = kl.TimeDistributed(cpg_model)(x)
        x = kl.GlobalAveragePooling1D()(x)

        return x
