"""Implements basic regularizers."""

from decaf import base
import numpy as np
from decaf.util import logexp


class RegularizationAsLossLayer(base.LossLayer):
    """This is a class that wraps around a specific regularizer class to
    create a loss layer. Different from the normal regularizer, which modifies
    a blob in-place and does not take into account the number of data points
    (which is desired in imposing regularization terms for parameters), the 
    wrapped layer divides the regularization output by the number of data
    points passed to the layer.

    kwargs:
        reg: the regularizer class.
        weight: the weight of the loss function.
        (You can add other parameters which your regularizer may need.)
    """
    def __init__(self, **kwargs):
        base.LossLayer.__init__(self, **kwargs)
        self._reg_kwargs = dict(self.spec)
        del self._reg_kwargs['reg']
        del self._reg_kwargs['weight']
        self._num_data = -1
        self._regularizer = None
    
    def _init_reg(self, num_data):
        if self._num_data != num_data:
            self._regularizer = self.spec['reg'](
                    weight=self.spec['weight'] / num_data, **self._reg_kwargs)
            self._num_data = num_data
    
    def forward(self, bottom, top):
        """Forward emits the loss, and computes the gradient as well."""
        num_data = bottom[0].data().shape[0]
        self._init_reg(num_data)
        diff = bottom[0].init_diff()
        self._loss = self._regularizer.reg(bottom[0])


def make_loss_layer_class(cls):
    def _make_layer(**kwargs):
        return RegularizationAsLossLayer(reg=cls, **kwargs)
    return _make_layer


# pylint: disable=R0903
class L2Regularizer(base.Regularizer):
    """The L2 regularization."""
    def reg(self, blob):
        """The reg function."""
        data = blob.data()
        #pylint: disable=W0612
        diff = blob.diff()
        diff += self._weight * 2. * data
        return np.dot(data.flat, data.flat) * self._weight

L2RegularizerLossLayer = make_loss_layer_class(L2Regularizer)


# pylint: disable=R0903
class L1Regularizer(base.Regularizer):
    """The L1 regularization."""
    def reg(self, blob):
        """The reg function."""
        data = blob.data()
        #pylint: disable=W0612
        diff = blob.diff()
        diff += self._weight * np.sign(data)
        return np.abs(data).sum() * self._weight


L1RegularizerLossLayer = make_loss_layer_class(L1Regularizer)
