"""Implements the dropout layer."""

from decaf import base
from decaf.layers import fillers
import numpy as np

class DropoutLayer(base.Layer):
    """A layer that implements the dropout.
    
    To increase test time efficiency, what we do in dropout is slightly
    different from the original version: instead of scaling during testing
    time, we scale up at training time so testing time is simply a mirroring
    operation.
    """

    def __init__(self, **kwargs):
        """Initializes a Dropout layer.

        kwargs:
            name: the layer name.
            ratio: the ratio to carry out dropout.
            debug_freeze: a debug flag. If set True, the mask will only
                be generated once when running. You should not use it other
                than purposes like gradient check.
        """
        base.Layer.__init__(self, **kwargs)
        filler = fillers.DropoutFiller(ratio=self.spec['ratio'])
        self._mask = base.Blob(filler=filler)

    def forward(self, bottom, top):
        """Computes the forward pass."""
        # Get features and output
        features = bottom[0].data()
        output = top[0].init_data(features.shape, features.dtype, setdata=False)
        if not self._mask.has_data():
            mask = self._mask.init_data(features.shape, np.bool)
        elif self.spec.get('debug_freeze', False):
            mask = self._mask.data()
        else:
            mask = self._mask.init_data(features.shape, np.bool)
        upscale = 1. / self.spec['ratio']
        output[:] = features * mask
        output *= upscale

    def predict(self, bottom, top):
        """The dropout predict pass. Under our definition, it is simply a
        mirror operation.
        """
        top[0].mirror(bottom[0])

    def backward(self, bottom, top, propagate_down):
        """Computes the backward pass."""
        if not propagate_down:
            return 0.
        top_diff = top[0].diff()
        bottom_diff = bottom[0].init_diff(setzero=False)
        mask = self._mask.data()
        upscale = 1. / self.spec['ratio']
        bottom_diff[:] = top_diff * mask
        bottom_diff *= upscale
        return 0.

    def update(self):
        """Dropout has nothing to update."""
        pass
