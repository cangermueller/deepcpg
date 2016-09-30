from keras import layers as kl
from keras import regularizers as kr

from .utils import Model


"""TODO
Residual block
Dilated convs
Strided conv instead of max pooling
Global average pooling
Try batch norm inputs
CpG as before or separate networks?
"""


class DnaModel(Model):

    def reader(self, data_files, *args, **kwargs):
        super_reader = super(DnaModel, self).reader
        for data in super_reader(data_files,
                                 use_dna=True,
                                 cpg_names=None,
                                 *args, **kwargs):
            yield data

    def inputs(self, dna_wlen):
        return [kl.Input(shape=(dna_wlen, 4), name='dna')]


class Dna01(DnaModel):

    def __call__(self, inputs):
        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        bn_axis = 2

        x = kl.Conv1D(64, 9, init='he_uniform', W_regularizer=w_reg)(inputs[0])
        x = kl.BatchNormalization(axis=bn_axis)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)
        x = kl.MaxPooling1D(2, 2)(x)

        x = kl.Conv1D(128, 3, init='he_uniform', W_regularizer=w_reg)(x)
        x = kl.BatchNormalization(axis=bn_axis)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)
        x = kl.MaxPooling1D(2, 2)(x)

        x = kl.Conv1D(256, 3, init='he_uniform', W_regularizer=w_reg)(x)
        x = kl.BatchNormalization(axis=bn_axis)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)
        x = kl.MaxPooling1D(2, 2)(x)

        x = kl.Conv1D(512, 3, init='he_uniform', W_regularizer=w_reg)(x)
        x = kl.BatchNormalization(axis=bn_axis)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)
        x = kl.MaxPooling1D(2, 2)(x)

        x = kl.GlobalAveragePooling1D()(x)

        return x
