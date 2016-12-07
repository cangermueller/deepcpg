from keras import layers as kl
from keras import regularizers as kr
from keras import models as km

from .utils import Model, encode_replicate_names
from ..utils import get_from_module


class CpgModel(Model):

    def __init__(self, *args, **kwargs):
        super(CpgModel, self).__init__(*args, **kwargs)
        self.scope = 'cpg'

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


class CpgAvg(CpgModel):
    """54000 params"""

    def _replicate_model(self, input):
        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(512, init=self.init, W_regularizer=w_reg)(input)
        x = kl.Activation('relu')(x)

        return km.Model(input, x)

    def __call__(self, inputs):
        x = self._merge_inputs(inputs)

        shape = getattr(x, '_keras_shape')
        replicate_model = self._replicate_model(kl.Input(shape=shape[2:]))
        x = kl.TimeDistributed(replicate_model)(x)
        x = kl.GlobalAveragePooling1D()(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)


class CpgRnn01(CpgModel):
    """810000 parameters"""

    def __init__(self, act_replicate='relu', *args, **kwargs):
        super(CpgRnn01, self).__init__(*args, **kwargs)
        self.act_replicate = act_replicate

    def _replicate_model(self, input):
        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(256, init=self.init, W_regularizer=w_reg)(input)
        x = kl.Activation(self.act_replicate)(x)

        return km.Model(input, x)

    def __call__(self, inputs):
        x = self._merge_inputs(inputs)

        shape = getattr(x, '_keras_shape')
        replicate_model = self._replicate_model(kl.Input(shape=shape[2:]))
        x = kl.TimeDistributed(replicate_model)(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Bidirectional(kl.GRU(256, W_regularizer=w_reg))(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)


class CpgRnn02(CpgRnn01):
    """810000 parameters"""

    def __init__(self, *args, **kwargs):
        super(CpgRnn02, self).__init__(act_replicate='linear', *args, **kwargs)


class CpgRnn03(CpgRnn01):
    """1112069 params"""

    def __call__(self, inputs):
        x = self._merge_inputs(inputs)

        shape = getattr(x, '_keras_shape')
        replicate_model = self._replicate_model(kl.Input(shape=shape[2:]))
        x = kl.TimeDistributed(replicate_model)(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Bidirectional(kl.GRU(128, W_regularizer=w_reg,
                                    return_sequences=True),
                             merge_mode='concat')(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Bidirectional(kl.GRU(256, W_regularizer=w_reg))(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)


class CpgRnn04(CpgRnn01):
    """810000 parameters"""

    def __init__(self, *args, **kwargs):
        super(CpgRnn04, self).__init__(act_replicate='tanh', *args, **kwargs)


def get(name):
    return get_from_module(name, globals())
