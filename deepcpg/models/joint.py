from keras import layers as kl
from keras import models as km
from keras import regularizers as kr

from .utils import Model

from ..utils import get_from_module


class JointModel(Model):

    def __init__(self, *args, **kwargs):
        super(JointModel, self).__init__(*args, **kwargs)
        self.mode = 'concat'
        self.scope = 'joint'

    def _get_inputs_outputs(self, models):
        inputs = []
        outputs = []
        for model in models:
            inputs.extend(model.inputs)
            outputs.extend(model.outputs)
        return (inputs, outputs)

    def _build(self, models, layers=[]):
        for layer in layers:
            layer.name = '%s/%s' % (self.scope, layer.name)

        inputs, outputs = self._get_inputs_outputs(models)
        x = kl.merge(outputs, mode=self.mode)
        for layer in layers:
            x = layer(x)

        model = km.Model(inputs, x, name=self.name)
        return model


class JointL0(JointModel):

    def __call__(self, models):
        return self._build(models)


class JointL1h512(JointModel):

    def __init__(self, nb_layer=1, nb_hidden=512, *args, **kwargs):
        super(JointL1h512, self).__init__(*args, **kwargs)
        self.nb_layer = nb_layer
        self.nb_hidden = nb_hidden

    def __call__(self, models):
        layers = []
        for layer in range(self.nb_layer):
            w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
            layers.append(kl.Dense(self.nb_hidden, init=self.init,
                                   W_regularizer=w_reg))
            layers.append(kl.Activation('relu'))
            layers.append(kl.Dropout(self.dropout))

        return self._build(models, layers)


class JointL2h512(JointL1h512):

    def __init__(self, *args, **kwargs):
        super(JointL2h512, self).__init__(*args, **kwargs)
        self.nb_layer = 2


class JointL3h512(JointL1h512):

    def __init__(self, *args, **kwargs):
        super(JointL3h512, self).__init__(*args, **kwargs)
        self.nb_layer = 3


def get(name):
    return get_from_module(name, globals())
