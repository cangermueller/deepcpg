from keras import layers as kl
from keras import models as km
from keras import regularizers as kr

from .utils import Model

from ..utils import get_from_module


class JointModel(Model):

    def _get_inputs_outputs(self, models):
        inputs = []
        outputs = []
        for model in models:
            inputs.extend(model.inputs)
            outputs.extend(model.outputs)

        return (inputs, outputs)


class Joint01(JointModel):

    def __init__(self, mode='concat', nb_hidden=1024, *args, **kwargs):
        super(JointModel, self).__init__(*args, **kwargs)
        self.mode = mode
        self.nb_hidden = nb_hidden

    def __call__(self, models):
        inputs, outputs = self._get_inputs_outputs(models)

        x = kl.merge(outputs, mode=self.mode)
        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg)(x)
        x = kl.BatchNormalization()(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        return km.Model(input=inputs, output=x, name=self.name)


class Joint02(JointModel):

    def __init__(self, mode='concat', *args, **kwargs):
        super(JointModel, self).__init__(*args, **kwargs)
        self.mode = mode

    def __call__(self, models):
        inputs, outputs = self._get_inputs_outputs(models)

        x = kl.merge(outputs, mode=self.mode)

        return km.Model(input=inputs, output=x, name=self.name)


def get(name):
    return get_from_module(name, globals())
