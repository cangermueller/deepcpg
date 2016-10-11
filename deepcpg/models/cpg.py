from keras import layers as kl
from keras import regularizers as kr
from keras import models as km

from .utils import Model


class CpgModel(Model):

    def __init__(self, replicate_names, cpg_wlen=None,
                 *args, **kwargs):
        super(CpgModel, self).__init__(*args, **kwargs)
        self.replicate_names = replicate_names
        self.cpg_wlen = cpg_wlen

    def reader(self, data_files, *args, **kwargs):
        super_reader = super(CpgModel, self).reader
        for data in super_reader(data_files,
                                 use_dna=False,
                                 replicate_names=self.replicate_names,
                                 cpg_wlen=self.cpg_wlen,
                                 *args, **kwargs):
            yield data

    def inputs(self):
        inputs = []
        shape = (len(self.replicate_names), self.cpg_wlen)
        inputs.append(kl.Input(shape=shape, name='cpg/state'))
        inputs.append(kl.Input(shape=shape, name='cpg/dist'))
        return inputs


class Cpg01(CpgModel):

    def _replicate_model(self, input):
        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(512, init='he_uniform', W_regularizer=w_reg)(input)
        x = kl.BatchNormalization(mode=2, axis=1)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)
        return km.Model(input=input, output=x)

    def __call__(self, inputs):
        x = kl.merge(inputs, mode='concat', concat_axis=2)

        x_shape = 2 * self.cpg_wlen
        replicate_model = self._replicate_model(kl.Input(shape=(x_shape,)))

        x = kl.TimeDistributed(replicate_model)(x)
        x = kl.GlobalAveragePooling1D()(x)

        return x


class Cpg02(CpgModel):

    def _replicate_model(self, input):
        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(256, init='he_uniform', W_regularizer=w_reg)(input)
        x = kl.BatchNormalization(mode=2, axis=1)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)
        return km.Model(input=input, output=x)

    def __call__(self, inputs):
        x = kl.merge(inputs, mode='concat', concat_axis=2)

        x_shape = 2 * self.cpg_wlen
        replicate_model = self._replicate_model(kl.Input(shape=(x_shape,)))

        x = kl.TimeDistributed(replicate_model)(x)
        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Bidirectional(kl.GRU(256, W_regularizer=w_reg))(x)

        return x


class Cpg03(CpgModel):

    def __call__(self, inputs):
        x = kl.merge(inputs, mode='concat', concat_axis=2)
        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Bidirectional(kl.GRU(256, W_regularizer=w_reg))(x)

        return x
