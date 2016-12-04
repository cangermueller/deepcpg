from keras import layers as kl
from keras import models as km
from keras import regularizers as kr

from .utils import Model

from ..utils import get_from_module


class JointModel(Model):

    def __init__(self, *args, **kwargs):
        super(JointModel, self).__init__(*args, **kwargs)
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


class Joint01(JointModel):

    def __init__(self, mode='concat', *args, **kwargs):
        super(Joint01, self).__init__(*args, **kwargs)
        self.mode = mode

    def __call__(self, models):
        return self._build(models)


class Joint02(JointModel):

    def __init__(self, mode='concat', nb_hidden=256, *args, **kwargs):
        super(Joint02, self).__init__(*args, **kwargs)
        self.mode = mode
        self.nb_hidden = nb_hidden

    def __call__(self, models):
        layers = []
        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        layers.append(kl.Dense(self.nb_hidden, init=self.init,
                               W_regularizer=w_reg))
        layers.append(kl.BatchNormalization())
        layers.append(kl.Activation('relu'))
        layers.append(kl.Dropout(self.dropout))

        return self._build(models, layers)


class Joint03(Joint02):

    def __init__(self, *args, **kwargs):
        super(Joint03, self).__init__(*args, **kwargs)
        self.nb_hidden = 512

def get(name):
    return get_from_module(name, globals())
