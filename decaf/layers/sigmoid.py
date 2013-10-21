"""Implements the sigmoid layer."""

from decaf import base
from decaf.util import logexp
import numexpr

class SigmoidLayer(base.Layer):
    """A layer that implements the sigmoid operation."""

    def __init__(self, **kwargs):
        """Initializes a ReLU layer.
        """
        base.Layer.__init__(self, **kwargs)
    
    def forward(self, bottom, top):
        """Computes the forward pass."""
        # Get features and top_data
        bottom_data = bottom[0].data()
        top_data = top[0].init_data(bottom_data.shape, bottom_data.dtype)
        numexpr.evaluate('1. / (exp(-bottom_data) + 1.)', out=top_data)

    def backward(self, bottom, top, propagate_down):
        """Computes the backward pass."""
        if propagate_down:
            top_data = top[0].data()
            top_diff = top[0].diff()
            bottom_diff = bottom[0].init_diff()
            numexpr.evaluate('top_data * top_diff * (1. - top_data)', out=bottom_diff)
        return 0

    def update(self):
        """ReLU has nothing to update."""
        pass
