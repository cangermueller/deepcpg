from keras import backend as K
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
        x = kl.BatchNormalization(mode=2, axis=1)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        return km.Model(input=input, output=x)

    def __call__(self, inputs=None):
        if inputs is None:
            inputs = self.inputs()

        x = self._merge_inputs(inputs)
        shape = getattr(x, '_keras_shape')
        replicate_model = self._replicate_model(kl.Input(shape=shape[2:]))
        x = kl.TimeDistributed(replicate_model)(x)
        x = kl.GlobalAveragePooling1D()(x)

        return km.Model(input=inputs, output=x, name=self.name)


class Cpg02(CpgModel):
    """FC 256 embedding layer. 814849 parameters"""

    def _replicate_model(self, input):
        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(256, init=self.init, W_regularizer=w_reg)(input)
        x = kl.BatchNormalization(mode=2, axis=1)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        return km.Model(input=input, output=x, name='replicate')

    def __call__(self, inputs):
        x = kl.merge(inputs, mode='concat', concat_axis=2)

        x_shape = 2 * self.cpg_wlen
        replicate_model = self._replicate_model(kl.Input(shape=(x_shape,)))

        x = kl.TimeDistributed(replicate_model)(x)
        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Bidirectional(kl.GRU(256, W_regularizer=w_reg))(x)

        return km.Model(input=inputs, output=x, name=self.name)


class Cpg03(CpgModel):
    """No FC embedding layer. 548865 parameters"""

    def __call__(self, inputs):
        x = kl.merge(inputs, mode='concat', concat_axis=2)
        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Bidirectional(kl.GRU(256, W_regularizer=w_reg))(x)

        return km.Model(input=inputs, output=x, name=self.name)


class Cpg04(CpgModel):
    """Old CNN"""

    def _replicate_model(self, input):
        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(41, 2, init='glorot_uniform', W_regularizer=w_reg)(input)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)
        x = kl.MaxPooling1D(2)(x)

        return km.Model(input=input, output=x, name='replicate')

    def __call__(self, inputs):
        replicate_input = kl.Input(shape=(self.cpg_wlen, 2,))
        replicate_model = self._replicate_model(replicate_input)

        x = []
        for input in inputs:
            x.append(K.expand_dims(input, 3))
        x = kl.merge(x, mode='concat', concat_axis=3)

        x = kl.TimeDistributed(replicate_model)(x)
        # N x C x 24 x 41
        shape = (len(self.replicate_names) * (self.cpg_wlen // 2 - 1) * 41,)
        x = kl.Reshape(shape)(x)

        return km.Model(input=inputs, output=x, name=self.name)


def get(name):
    return get_from_module(name, globals())
