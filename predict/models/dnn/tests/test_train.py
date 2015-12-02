import numpy as np
import numpy.testing as npt

from predict.models.dnn.train import sample_weights


def test_sample_weights():
    y = np.array([1, -1, 0, 1, -1])

    w = sample_weights(y)
    npt.assert_array_equal(w, [1, 0, 1, 1, 0])

    w = sample_weights(y, weight_classes=True)
    t = 1 / 3
    npt.assert_array_almost_equal(w, [t, 0, 1 - t, t, 0], 3)

