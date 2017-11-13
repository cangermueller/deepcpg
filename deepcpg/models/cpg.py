"""CpG models.

Provides models trained with observed neighboring methylation states of
multiple cells.
"""

from __future__ import division
from __future__ import print_function

import inspect

from keras import layers as kl
from keras import regularizers as kr
from keras import models as km
from keras.layers.merge import concatenate

from .utils import Model
from ..utils import get_from_module


class CpgModel(Model):
    """Abstract class of a CpG model."""

    def __init__(self, *args, **kwargs):
        super(CpgModel, self).__init__(*args, **kwargs)
        self.scope = 'cpg'

    def inputs(self, cpg_wlen, replicate_names):
        inputs = []
        shape = (len(replicate_names), cpg_wlen)
        inputs.append(kl.Input(shape=shape, name='cpg/state'))
        inputs.append(kl.Input(shape=shape, name='cpg/dist'))
        return inputs

    def _merge_inputs(self, inputs):
        return concatenate(inputs, axis=2)


class FcAvg(CpgModel):
    """Fully-connected layer followed by global average layer.

    .. code::

        Parameters: 54,000
        Specification: fc[512]_gap
    """

    def _replicate_model(self, input):
        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(512, kernel_initializer=self.init,
                     kernel_regularizer=kernel_regularizer)(input)
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


class RnnL1(CpgModel):
    """Bidirectional GRU with one layer.

    .. code::

        Parameters: 810,000
        Specification: fc[256]_bgru[256]_do
    """

    def __init__(self, act_replicate='relu', *args, **kwargs):
        super(RnnL1, self).__init__(*args, **kwargs)
        self.act_replicate = act_replicate

    def _replicate_model(self, input):
        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(256, kernel_initializer=self.init,
                     kernel_regularizer=kernel_regularizer)(input)
        x = kl.Activation(self.act_replicate)(x)

        return km.Model(input, x)

    def __call__(self, inputs):
        x = self._merge_inputs(inputs)

        shape = getattr(x, '_keras_shape')
        replicate_model = self._replicate_model(kl.Input(shape=shape[2:]))
        x = kl.TimeDistributed(replicate_model)(x)

        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        gru = kl.GRU(256, kernel_regularizer=kernel_regularizer)
        x = kl.Bidirectional(gru)(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)


class RnnL2(RnnL1):
    """Bidirectional GRU with two layers.

    .. code::

        Parameters: 1,100,000
        Specification: fc[256]_bgru[128]_bgru[256]_do
    """

    def __call__(self, inputs):
        x = self._merge_inputs(inputs)

        shape = getattr(x, '_keras_shape')
        replicate_model = self._replicate_model(kl.Input(shape=shape[2:]))
        x = kl.TimeDistributed(replicate_model)(x)

        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Bidirectional(kl.GRU(128, kernel_regularizer=kernel_regularizer,
                                    return_sequences=True),
                             merge_mode='concat')(x)

        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        gru = kl.GRU(256, kernel_regularizer=kernel_regularizer)
        x = kl.Bidirectional(gru)(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)


def list_models():
    """Return the name of models in the module."""

    models = dict()
    for name, value in globals().items():
        if inspect.isclass(value) and name.lower().find('model') == -1:
            models[name] = value
    return models


def get(name):
    """Return object from module by its name."""
    return get_from_module(name, globals())
