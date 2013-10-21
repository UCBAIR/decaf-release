"""Implements the ReLU layer."""

from decaf import base
from decaf.layers.cpp import wrapper

class ReLULayer(base.Layer):
    """A layer that implements the Regularized Linear Unit (ReLU) operation
    that converts x to max(x, 0).
    """

    def __init__(self, **kwargs):
        """Initializes a ReLU layer.
        """
        base.Layer.__init__(self, **kwargs)
    
    def forward(self, bottom, top):
        """Computes the forward pass."""
        # Get features and output
        features = bottom[0].data()
        output = top[0].init_data(features.shape, features.dtype)
        wrapper.relu_forward(features, output)
        #output[:] = features
        #output *= (features > 0)

    def backward(self, bottom, top, propagate_down):
        """Computes the backward pass."""
        if not propagate_down:
            return 0.
        top_diff = top[0].diff()
        features = bottom[0].data()
        bottom_diff = bottom[0].init_diff()
        bottom_diff[:] = top_diff
        bottom_diff *= (features > 0)
        return 0.

    def update(self):
        """ReLU has nothing to update."""
        pass
