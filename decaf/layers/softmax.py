"""Implements the softmax function.
"""

from decaf import base
from decaf.util import logexp
import numpy as np
from numpy.core.umath_tests import inner1d

class SoftmaxLayer(base.Layer):
    """A layer that implements the softmax function."""

    def __init__(self, **kwargs):
        """Initializes a softmax layer.

        kwargs:
            name: the layer name.
        """
        base.Layer.__init__(self, **kwargs)

    def forward(self, bottom, top):
        """Computes the forward pass."""
        # Get features and output
        pred = bottom[0].data()
        prob = top[0].init_data(pred.shape, pred.dtype, setdata=False)
        prob[:] = pred
        # normalize by subtracting the max to suppress numerical issues
        prob -= prob.max(axis=1)[:, np.newaxis]
        logexp.exp(prob, out=prob)
        prob /= prob.sum(axis=1)[:, np.newaxis]

    def backward(self, bottom, top, propagate_down):
        """Computes the backward pass."""
        if not propagate_down:
            return 0.
        top_diff = top[0].diff()
        prob = top[0].data()
        bottom_diff = bottom[0].init_diff(setzero=False)
        bottom_diff[:] = top_diff
        cross_term = inner1d(top_diff, prob)
        bottom_diff -= cross_term[:, np.newaxis]
        bottom_diff *= prob
        return 0.

    def update(self):
        """Softmax has nothing to update."""
        pass
