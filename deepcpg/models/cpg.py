from keras import layers as kl
from keras import regularizers as kr
from keras import models as km

from .utils import Model, encode_replicate_names
from ..utils import get_from_module


class CpgModel(Model):

    def inputs(self, cpg_wlen, replicate_names):
        inputs = []
        shape = (len(replicate_names), cpg_wlen)
        replicates_id = encode_replicate_names(replicate_names)
        inputs.append(kl.Input(shape=shape,
                               name='cpg/state/%s' % replicates_id))
        inputs.append(kl.Input(shape=shape,
                               name='cpg/dist/%s' % replicates_id))
        return inputs

    def _merge_inputs(self, inputs):
        return kl.merge(inputs, mode='concat', concat_axis=2)


class Cpg01(CpgModel):

    def _replicate_model(self, input):
        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(512, init=self.init, W_regularizer=w_reg)(input)
        x = kl.Activation('relu')(x)

        return km.Model(input=input, output=x)

    def __call__(self, inputs):
        x = self._merge_inputs(inputs)

        shape = getattr(x, '_keras_shape')
        replicate_model = self._replicate_model(kl.Input(shape=shape[2:]))
        x = kl.TimeDistributed(replicate_model)(x)
        x = kl.GlobalAveragePooling1D()(x)
        x = kl.Dropout(self.dropout)(x)

        return km.Model(input=inputs, output=x, name=self.name)


class Cpg02(CpgModel):
    """GRU(256) without embedding layer; 548865 parameters"""

    def __call__(self, inputs):
        x = kl.merge(inputs, mode='concat', concat_axis=2)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Bidirectional(kl.GRU(256, W_regularizer=w_reg))(x)
        x = kl.Dropout(self.dropout)(x)

        return km.Model(input=inputs, output=x, name=self.name)


class Cpg03(CpgModel):
    """GRU(256) with embedding layer; 814849 parameters"""

    def _replicate_model(self, input):
        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(256, init=self.init, W_regularizer=w_reg)(input)
        x = kl.Activation('relu')(x)

        return km.Model(input=input, output=x)

    def __call__(self, inputs):
        x = self._merge_inputs(inputs)

        shape = getattr(x, '_keras_shape')
        replicate_model = self._replicate_model(kl.Input(shape=shape[2:]))
        x = kl.TimeDistributed(replicate_model)(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Bidirectional(kl.GRU(256, W_regularizer=w_reg))(x)
        x = kl.Dropout(self.dropout)(x)

        return km.Model(input=inputs, output=x, name=self.name)


class Cpg04(CpgModel):
    """GRU with two layers (2x128, 2x256) with embedding layer.
    964353 parameters"""

    def __call__(self, inputs):
        x = self._merge_inputs(inputs)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Bidirectional(kl.GRU(128, W_regularizer=w_reg,
                                    return_sequences=True),
                             merge_mode='concat')(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Bidirectional(kl.GRU(256, W_regularizer=w_reg))(x)
        x = kl.Dropout(self.dropout)(x)

        return km.Model(input=inputs, output=x, name=self.name)


def get(name):
    return get_from_module(name, globals())
