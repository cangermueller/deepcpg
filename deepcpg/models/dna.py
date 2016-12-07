from keras import layers as kl
from keras import regularizers as kr

from .utils import Model
from ..utils import get_from_module


class DnaModel(Model):

    def __init__(self, *args, **kwargs):
        super(DnaModel, self).__init__(*args, **kwargs)
        self.scope = 'dna'

    def inputs(self, dna_wlen):
        return [kl.Input(shape=(dna_wlen, 4), name='dna')]


class DnaLegacy(DnaModel):
    """Old DNA module"""

    def __call__(self, inputs):
        x = inputs[0]

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(97, 12, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(3)(x)
        x = kl.Dropout(self.dropout)(x)

        x = kl.Flatten()(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(1024, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)


class DnaL1_01(DnaModel):
    """default"""

    def __call__(self, inputs):
        x = inputs[0]

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(128, 11, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        x = kl.Flatten()(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(128, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)


class DnaL1_02(DnaModel):
    """he_normal"""

    def __init__(self, *args, **kwargs):
        super(DnaL1_02, self).__init__(*args, **kwargs)
        self.init = 'he_normal'

    def __call__(self, inputs):
        x = inputs[0]

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(128, 9, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        x = kl.Flatten()(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(128, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)


class DnaL1_03(DnaModel):
    """256"""

    def __call__(self, inputs):
        x = inputs[0]

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(128, 11, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        x = kl.Flatten()(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(256, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)


class DnaL1_04(DnaModel):
    """batch norm"""

    def __call__(self, inputs):
        x = inputs[0]

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(128, 11, W_regularizer=w_reg)(x)
        x = kl.BatchNormalization()(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        x = kl.Flatten()(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(128, init=self.init, W_regularizer=w_reg)(x)
        x = kl.BatchNormalization()(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)


class DnaL2_01(DnaModel):
    """two layers; 4.100.000"""

    def __call__(self, inputs):
        x = inputs[0]

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(128, 11, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(256, 3, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(2)(x)

        x = kl.Flatten()(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(128, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)


class DnaL2_02(DnaModel):
    """two layers; batch-norm"""

    def __call__(self, inputs):
        x = inputs[0]

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(128, 11, init=self.init, W_regularizer=w_reg)(x)
        x = kl.BatchNormalization()(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(256, 3, init=self.init, W_regularizer=w_reg)(x)
        x = kl.BatchNormalization()(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(2)(x)

        x = kl.Flatten()(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(128, init=self.init, W_regularizer=w_reg)(x)
        x = kl.BatchNormalization()(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)


class DnaL2_03(DnaModel):
    """global average"""

    def __call__(self, inputs):
        x = inputs[0]

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(128, 11, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(256, 3, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(2)(x)

        x = kl.GlobalAveragePooling1D()(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(512, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)


class DnaL3_01(DnaModel):
    """three layers; flatten; 4.400.000 params"""

    def __call__(self, inputs):
        x = inputs[0]

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(128, 11, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(256, 3, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(2)(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(512, 3, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(2)(x)

        x = kl.Flatten()(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(128, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)


class DnaL3_02(DnaModel):
    """three layers; global avg; 760.000 params"""

    def __call__(self, inputs):
        x = inputs[0]

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(128, 11, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(256, 3, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(2)(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(512, 3, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(2)(x)

        x = kl.GlobalAveragePooling1D()(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(512, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)


class Dna01(DnaModel):
    """Simple: 126785 params"""

    def __call__(self, inputs):
        x = inputs[0]

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(64, 9, init=self.init, W_regularizer=w_reg)(x)
        x = kl.BatchNormalization()(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(128, 3, init=self.init, W_regularizer=w_reg)(x)
        x = kl.BatchNormalization()(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(256, 3, init=self.init, W_regularizer=w_reg)(x)
        x = kl.BatchNormalization()(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(2)(x)

        x = kl.GlobalAveragePooling1D()(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)


class CnnRnn01(DnaModel):
    """CNN + RNN: 1139457 params"""

    def __call__(self, inputs):
        x = inputs[0]

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(128, 9, init=self.init, W_regularizer=w_reg)(x)
        x = kl.BatchNormalization()(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(2)(x)
        # 250

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(128, 3, init=self.init, W_regularizer=w_reg)(x)
        x = kl.BatchNormalization()(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(2)(x)
        # 128

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(256, 3, init=self.init, W_regularizer=w_reg)(x)
        x = kl.BatchNormalization()(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(2)(x)
        # 62

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(256, 3, init=self.init, W_regularizer=w_reg)(x)
        x = kl.BatchNormalization()(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(2)(x)
        # 32

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Bidirectional(kl.recurrent.GRU(256,
                                              return_sequences=False,
                                              W_regularizer=w_reg))(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)


class ResNet01(DnaModel):
    "1745281 parameters"

    def _res_block(self, inputs, nb_filter, size=3, stride=1, stage=1, block=1):

        name = '%02d-%02d/' % (stage, block)
        id_name = '%sid_' % (name)
        res_name = '%sres_' % (name)

        # Residual branch

        # 1x1 down-sample conv
        x = kl.BatchNormalization(name=res_name + 'bn1')(inputs)
        x = kl.Activation('relu', name=res_name + 'act1')(x)
        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(nb_filter[0], 1,
                      name=res_name + 'conv1',
                      subsample_length=stride,
                      init=self.init,
                      W_regularizer=w_reg)(x)

        # LxL conv
        x = kl.BatchNormalization(name=res_name + 'bn2')(x)
        x = kl.Activation('relu', name=res_name + 'act2')(x)
        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(nb_filter[1], size,
                      name=res_name + 'conv2',
                      border_mode='same',
                      init=self.init,
                      W_regularizer=w_reg)(x)

        # 1x1 up-sample conv
        x = kl.BatchNormalization(name=res_name + 'bn3')(x)
        x = kl.Activation('relu', name=res_name + 'act3')(x)
        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(nb_filter[2], 1,
                      name=res_name + 'conv3',
                      init=self.init,
                      W_regularizer=w_reg)(x)

        # Identity branch
        if nb_filter[-1] != inputs._keras_shape[-1] or stride > 1:
            w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
            identity = kl.Conv1D(nb_filter[2], 1,
                                 name=id_name + 'conv1',
                                 subsample_length=stride,
                                 init=self.init,
                                 W_regularizer=w_reg)(inputs)
        else:
            identity = inputs

        x = kl.merge([identity, x], name=name + 'merge', mode='sum')

        return x

    def __call__(self, inputs):
        x = inputs[0]

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(128, 11,
                      name='conv1',
                      init=self.init,
                      W_regularizer=w_reg)(x)
        x = kl.Activation('relu', name='act1')(x)
        x = kl.MaxPooling1D(2, name='pool1')(x)

        # 124
        x = self._res_block(x, [32, 32, 128], stage=1, block=1, stride=2)
        x = self._res_block(x, [32, 32, 128], stage=1, block=2)

        # 64
        x = self._res_block(x, [64, 64, 256], stage=2, block=1, stride=2)
        x = self._res_block(x, [64, 64, 256], stage=2, block=2)

        # 32
        x = self._res_block(x, [128, 128, 512], stage=3, block=1, stride=2)
        x = self._res_block(x, [128, 128, 512], stage=3, block=2)

        # 16
        x = self._res_block(x, [256, 256, 1024], stage=4, block=1, stride=2)

        x = kl.GlobalAveragePooling1D()(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)


class ResNet02(ResNet01):
    "Like ResNet01, but more blocks per stage.  1985857 parameters"

    def __call__(self, inputs):
        x = inputs[0]

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(128, 11,
                      name='conv1',
                      init=self.init,
                      W_regularizer=w_reg)(x)
        x = kl.BatchNormalization(name='bn1')(x)
        x = kl.Activation('relu', name='act1')(x)
        x = kl.MaxPooling1D(2, name='pool1')(x)

        # 124
        x = self._res_block(x, [32, 32, 128], stage=1, block=1, stride=2)
        x = self._res_block(x, [32, 32, 128], stage=1, block=2)
        x = self._res_block(x, [32, 32, 128], stage=1, block=3)

        # 64
        x = self._res_block(x, [64, 64, 256], stage=2, block=1, stride=2)
        x = self._res_block(x, [64, 64, 256], stage=2, block=2)
        x = self._res_block(x, [64, 64, 256], stage=2, block=3)

        # 32
        x = self._res_block(x, [128, 128, 512], stage=3, block=1, stride=2)
        x = self._res_block(x, [128, 128, 512], stage=3, block=2)
        x = self._res_block(x, [128, 128, 512], stage=3, block=3)

        # 16
        x = self._res_block(x, [256, 256, 1024], stage=4, block=1, stride=2)

        x = kl.GlobalAveragePooling1D()(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)


class ResNet03(ResNet01):

    def __init__(self, *args, **kwargs):
        super(ResNet03, self).__init__(*args, **kwargs)
        self.init = 'he_normal'


class ResConv01(ResNet01):
    "2 conv instead of bottleneck. 2815233 parameters"

    def _res_block(self, inputs, nb_filter, size=3, stride=1, stage=1, block=1):

        name = '%02d-%02d/' % (stage, block)
        id_name = '%sid_' % (name)
        res_name = '%sres_' % (name)

        # Residual branch
        x = kl.BatchNormalization(name=res_name + 'bn1')(inputs)
        x = kl.Activation('relu', name=res_name + 'act1')(x)
        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(nb_filter, size,
                      name=res_name + 'conv1',
                      border_mode='same',
                      subsample_length=stride,
                      init=self.init,
                      W_regularizer=w_reg)(x)

        x = kl.BatchNormalization(name=res_name + 'bn2')(x)
        x = kl.Activation('relu', name=res_name + 'act2')(x)
        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(nb_filter, size,
                      name=res_name + 'conv2',
                      border_mode='same',
                      init=self.init,
                      W_regularizer=w_reg)(x)

        # Identity branch
        if nb_filter != inputs._keras_shape[-1] or stride > 1:
            w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
            identity = kl.Conv1D(nb_filter, size,
                                 name=id_name + 'conv1',
                                 border_mode='same',
                                 subsample_length=stride,
                                 init=self.init,
                                 W_regularizer=w_reg)(inputs)
        else:
            identity = inputs

        x = kl.merge([identity, x], name=name + 'merge', mode='sum')

        return x

    def __call__(self, inputs):
        x = inputs[0]

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(128, 9,
                      name='conv1',
                      init=self.init,
                      W_regularizer=w_reg)(x)
        x = kl.BatchNormalization(name='bn1')(x)
        x = kl.Activation('relu', name='act1')(x)
        x = kl.MaxPooling1D(2, name='pool1')(x)

        # 124
        x = self._res_block(x, 128, stage=1, block=1, stride=2)
        x = self._res_block(x, 128, stage=1, block=2)

        # 64
        x = self._res_block(x, 256, stage=2, block=1, stride=2)

        # 32
        x = self._res_block(x, 256, stage=3, block=1, stride=2)

        # 32
        x = self._res_block(x, 512, stage=4, block=1, stride=2)

        x = kl.GlobalAveragePooling1D()(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)


class ResAtrous01(DnaModel):
    """1989957 params"""

    def _res_block(self, inputs, nb_filter, size=3, stride=1, atrous=1,
                   stage=1, block=1):

        name = '%02d-%02d/' % (stage, block)
        id_name = '%sid_' % (name)
        res_name = '%sres_' % (name)

        # Residual branch

        # 1x1 down-sample conv
        x = kl.BatchNormalization(name=res_name + 'bn1')(inputs)
        x = kl.Activation('relu', name=res_name + 'act1')(x)
        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(nb_filter[0], 1,
                      name=res_name + 'conv1',
                      subsample_length=stride,
                      init=self.init,
                      W_regularizer=w_reg)(x)

        # LxL conv
        x = kl.BatchNormalization(name=res_name + 'bn2')(x)
        x = kl.Activation('relu', name=res_name + 'act2')(x)
        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.AtrousConv1D(nb_filter[1], size,
                            atrous_rate=atrous,
                            name=res_name + 'conv2',
                            border_mode='same',
                            init=self.init,
                            W_regularizer=w_reg)(x)

        # 1x1 up-sample conv
        x = kl.BatchNormalization(name=res_name + 'bn3')(x)
        x = kl.Activation('relu', name=res_name + 'act3')(x)
        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(nb_filter[2], 1,
                      name=res_name + 'conv3',
                      init=self.init,
                      W_regularizer=w_reg)(x)

        # Identity branch
        if nb_filter[-1] != inputs._keras_shape[-1] or stride > 1:
            w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
            identity = kl.Conv1D(nb_filter[2], 1,
                                 name=id_name + 'conv1',
                                 subsample_length=stride,
                                 init=self.init,
                                 W_regularizer=w_reg)(inputs)
        else:
            identity = inputs

        x = kl.merge([identity, x], name=name + 'merge', mode='sum')

        return x

    def __call__(self, inputs):
        x = inputs[0]

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(128, 9,
                      name='conv1',
                      init=self.init,
                      W_regularizer=w_reg)(x)
        x = kl.Activation('relu', name='act1')(x)
        x = kl.MaxPooling1D(2, name='pool1')(x)

        # 124
        x = self._res_block(x, [32, 32, 128], stage=1, block=1, stride=2)
        x = self._res_block(x, [32, 32, 128], atrous=2, stage=1, block=2)
        x = self._res_block(x, [32, 32, 128], atrous=4, stage=1, block=3)

        # 64
        x = self._res_block(x, [64, 64, 256], stage=2, block=1, stride=2)
        x = self._res_block(x, [64, 64, 256], atrous=2, stage=2, block=2)
        x = self._res_block(x, [64, 64, 256], atrous=4, stage=2, block=3)

        # 32
        x = self._res_block(x, [128, 128, 512], stage=3, block=1, stride=2)
        x = self._res_block(x, [128, 128, 512], atrous=2, stage=3, block=2)
        x = self._res_block(x, [128, 128, 512], atrous=4, stage=3, block=3)

        # 16
        x = self._res_block(x, [256, 256, 1024], stage=4, block=1, stride=2)

        x = kl.GlobalAveragePooling1D()(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)


def get(name):
    return get_from_module(name, globals())
