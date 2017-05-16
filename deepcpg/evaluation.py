"""Functions for evaluating prediction performance."""

from __future__ import division
from __future__ import print_function

from collections import OrderedDict

import numpy as np
import pandas as pd
import sklearn.metrics as skm
from scipy.stats import kendalltau
from six.moves import range

from .data import CPG_NAN, OUTPUT_SEP
from .utils import get_from_module


def cor(y, z):
    """Compute Pearson's correlation coefficient."""
    return np.corrcoef(y, z)[0, 1]


def kendall(y, z, nb_sample=100000):
    """Compute Kendall's correlation coefficient."""
    if len(y) > nb_sample:
        idx = np.arange(len(y))
        np.random.shuffle(idx)
        idx = idx[:nb_sample]
        y = y[idx]
        z = z[idx]
    return kendalltau(y, z)[0]


def mad(y, z):
    """Compute mean absolute deviation."""
    return np.mean(np.abs(y - z))


def mse(y, z):
    """Compute mean squared error."""
    return np.mean((y - z)**2)


def rmse(y, z):
    """Compute root mean squared error."""
    return np.sqrt(mse(y, z))


def auc(y, z, round=True):
    """Compute area under the ROC curve."""
    if round:
        y = y.round()
    if len(y) == 0 or len(np.unique(y)) < 2:
        return np.nan
    return skm.roc_auc_score(y, z)


def acc(y, z, round=True):
    """Compute accuracy."""
    if round:
        y = np.round(y)
        z = np.round(z)
    return skm.accuracy_score(y, z)


def tpr(y, z, round=True):
    """Compute true positive rate."""
    if round:
        y = np.round(y)
        z = np.round(z)
    return skm.recall_score(y, z)


def tnr(y, z, round=True):
    """Compute true negative rate."""
    if round:
        y = np.round(y)
        z = np.round(z)
    c = skm.confusion_matrix(y, z)
    return c[0, 0] / c[0].sum()


def mcc(y, z, round=True):
    """Compute Matthew's correlation coefficient."""
    if round:
        y = np.round(y)
        z = np.round(z)
    return skm.matthews_corrcoef(y, z)


def f1(y, z, round=True):
    """Compute F1 score."""
    if round:
        y = np.round(y)
        z = np.round(z)
    return skm.f1_score(y, z)


def cat_acc(y, z):
    """Compute categorical accuracy given one-hot matrices."""
    return np.mean(y.argmax(axis=1) == z.argmax(axis=1))


# Classification metrics.
CLA_METRICS = [auc, acc, tpr, tnr, f1, mcc]

# Regression metrics.
REG_METRICS = [mse, mad, cor]

# Categorical metrics.
CAT_METRICS = [cat_acc]


def evaluate(y, z, mask=CPG_NAN, metrics=CLA_METRICS):
    """Compute multiple performance metrics.

    Computes evaluation metrics using functions in `metrics`.

    Parameters
    ----------
    y: :class:`numpy.ndarray`
        :class:`numpy.ndarray` vector with labels.
    z: :class:`numpy.ndarray`
        :class:`numpy.ndarray` vector with predictions.
    mask: scalar
        Value to mask unobserved labels in `y`.
    metrics: list
        List of evaluation functions to be used.

    Returns
    -------
    Ordered dict
        Ordered dict with name of evaluation functions as keys and evaluation
        metrics as values.
    """
    z = z.ravel()
    if mask is not None:
        t = y != mask
        y = y[t]
        z = z[t]
    p = OrderedDict()
    for metric in metrics:
        if len(y):
            p[metric.__name__] = metric(y, z)
        else:
            p[metric.__name__] = np.nan
    p['n'] = len(y)
    return p


def evaluate_cat(y, z, metrics=CAT_METRICS,
                 binary_metrics=None):
    """Compute multiple performance metrics for categorical outputs.

    Computes evaluation metrics for categorical (one-hot encoded labels) using
    functions in `metrics`.

    Parameters
    ----------
    y: :class:`numpy.ndarray`
        :class:`numpy.ndarray` matrix with one-hot encoded labels.
    z: :class:`numpy.ndarray`
        :class:`numpy.ndarray` matrix with class probabilities in rows.
    metrics: list
        List of evaluation functions to be used.
    binary_metrics: list
        List of binary evaluation metrics to be computed for each category, e.g.
        class, separately. Will be encoded as `name_i` in the output dictionary,
        where `name` is the name of the evaluation metrics and `i` the index of
        the category.

    Returns
    -------
    Ordered dict
        Ordered dict with name of evaluation functions as keys and evaluation
        metrics as values.
    """
    idx = y.sum(axis=1) > 0
    y = y[idx]
    z = z[idx]
    p = OrderedDict()
    for metric in metrics:
        p[metric.__name__] = metric(y, z)
    if binary_metrics:
        for i in range(y.shape[1]):
            for metric in binary_metrics:
                p['%s_%d' % (metric.__name__, i)] = metric(y[:, i], z[:, i])
    p['n'] = len(y)
    return p


def get_output_metrics(output_name):
    """Return list of evaluation metrics for model output name."""
    _output_name = output_name.split(OUTPUT_SEP)
    if _output_name[0] == 'cpg':
        metrics = CLA_METRICS
    elif _output_name[0] == 'bulk':
        metrics = REG_METRICS + CLA_METRICS
    elif _output_name[-1] in ['diff', 'mode', 'cat2_var']:
        metrics = CLA_METRICS
    elif _output_name[-1] == 'mean':
        metrics = REG_METRICS + CLA_METRICS + [kendall]
    elif _output_name[-1] == 'var':
        metrics = REG_METRICS + [kendall]
    else:
        raise ValueError('Invalid output name "%s"!' % output_name)
    return metrics


def evaluate_outputs(outputs, preds):
    """Evaluate performance metrics of multiple outputs.

    Given the labels and predictions of multiple outputs, chooses and computes
    performance metrics of each output depending on its name.

    Parameters
    ----------
    outputs: dict
        `dict` with the name of outputs as keys and a :class:`numpy.ndarray`
        vector with labels as value.
    preds: dict
        `dict` with the name of outputs as keys and a :class:`numpy.ndarray`
        vector with predictions as value.

    Returns
    -------
    :class:`pandas.DataFrame`
        :class:`pandas.DataFrame` with columns `metric`, `output`, `value`.
    """
    perf = []
    for output_name in outputs:
        _output_name = output_name.split(OUTPUT_SEP)
        if _output_name[-1] in ['cat_var']:
            tmp = evaluate_cat(outputs[output_name],
                               preds[output_name],
                               binary_metrics=[auc])
        else:
            metrics = get_output_metrics(output_name)
            tmp = evaluate(outputs[output_name],
                           preds[output_name],
                           metrics=metrics)
        tmp = pd.DataFrame({'output': output_name,
                            'metric': list(tmp.keys()),
                            'value': list(tmp.values())})
        perf.append(tmp)
    perf = pd.concat(perf)
    perf = perf[['metric', 'output', 'value']]
    perf.sort_values(['metric', 'value'], inplace=True)
    return perf


def is_binary_output(output_name):
    """Return `True` if `output_name` is binary."""
    _output_name = output_name.split(OUTPUT_SEP)
    if _output_name[0] == 'cpg':
        return True
    elif _output_name[-1] in ['diff', 'mode', 'cat2_var']:
        return True
    else:
        return False


def evaluate_curve(outputs, preds, fun=skm.roc_curve, mask=CPG_NAN,
                   nb_point=None):
    """Evaluate performance curves of multiple outputs.

    Given the labels and predictions of multiple outputs, computes a performance
    a curve, e.g. ROC or PR curve, for each output.

    Parameters
    ----------
    outputs: dict
        `dict` with the name of outputs as keys and a :class:`numpy.ndarray`
        vector with labels as value.
    preds: dict
        `dict` with the name of outputs as keys and a :class:`numpy.ndarray`
        vector with predictions as value.
    fun: function
        Function to compute the performance curves.
    mask: scalar
        Value to mask unobserved labels in `y`.
    nb_point: int
        Maximum number of points to curve to reduce memory.

    Returns
    -------
    :class:`pandas.DataFrame`
        :class:`pandas.DataFrame` with columns `output`, `x`, `y`, `thr`.
    """
    curves = []
    for output_name in outputs.keys():
        if not is_binary_output(output_name):
            continue

        output = outputs[output_name].round().squeeze()
        pred = preds[output_name].squeeze()
        idx = output != CPG_NAN
        output = output[idx]
        pred = pred[idx]

        x, y, thr = fun(output, pred)
        length = min(len(x), len(y), len(thr))
        if nb_point and length > nb_point:
            idx = np.linspace(0, length - 1, nb_point).astype(np.int32)
        else:
            idx = slice(0, length)
        x = x[idx]
        y = y[idx]
        thr = thr[idx]

        curve = OrderedDict()
        curve['output'] = output_name
        curve['x'] = x
        curve['y'] = y
        curve['thr'] = thr
        curve = pd.DataFrame(curve)
        curves.append(curve)

    if not curves:
        return None
    else:
        curves = pd.concat(curves)
        return curves


def unstack_report(report):
    """Unstack performance report.

    Reshapes a :class:`pandas.DataFrame` of :func:`evaluate_outputs` such that
    performance metrics are listed as columns.

    Parameters
    ----------
    report: :class:`pandas.DataFrame`
        :class:`pandas.DataFrame` from :func:`evaluate_outputs`.

    Returns
    -------
    :class:`pandas.DataFrame`
        :class:`pandas.DataFrame` with performance metrics as columns.
    """
    index = list(report.columns[~report.columns.isin(['metric', 'value'])])
    report = pd.pivot_table(report, index=index, columns='metric',
                            values='value')
    report.reset_index(index, inplace=True)
    report.columns.name = None

    # Sort columns
    columns = list(report.columns)
    sorted_columns = []
    for fun in CAT_METRICS + CLA_METRICS + REG_METRICS:
        for i, column in enumerate(columns):
            if column.startswith(fun.__name__):
                sorted_columns.append(column)
    sorted_columns = index + sorted_columns
    sorted_columns += [col for col in columns if col not in sorted_columns]
    report = report[sorted_columns]
    order = []
    if 'auc' in report.columns:
        order.append(('auc', False))
    elif 'mse' in report.columns:
        order.append(('mse', True))
    elif 'acc' in report.columns:
        order.append(('acc', False))
    report.sort_values([x[0] for x in order],
                       ascending=[x[1] for x in order],
                       inplace=True)
    return report


def get(name):
    """Return object from module by its name."""
    return get_from_module(name, globals())
