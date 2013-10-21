"""Implements the inner product layer."""

from decaf import base
from decaf.util import blasdot
import numpy as np

class InnerProductLayer(base.Layer):
    """A layer that implements the inner product."""

    def __init__(self, **kwargs):
        """Initializes an inner product layer. 
        
        kwargs:
            num_output: the number of outputs.
            reg: the regularizer to be used to add regularization terms.
                should be a decaf.base.Regularizer instance. Default None. 
            filler: a filler to initialize the weights. Should be a
                decaf.base.Filler instance. Default None.
            bias_filler: a filler to initialize the bias.
            bias: if True, the inner product will contain a bias term.
                Default True.
        """
        base.Layer.__init__(self, **kwargs)
        self._num_output = self.spec.get('num_output', 0)
        if self._num_output <= 0:
            raise base.InvalidLayerError(
                'Incorrect or unspecified num_output for %s' % self.name)
        self._reg = self.spec.get('reg', None)
        self._filler = self.spec.get('filler', None)
        self._weight = base.Blob(filler=self._filler)
        self._has_bias = self.spec.get('bias', True)
        if self._has_bias:
            self._bias_filler = self.spec.get('bias_filler', None)
            self._bias = base.Blob(filler=self._bias_filler)
            self._param = [self._weight, self._bias]
        else:
            self._param = [self._weight]
    
    def forward(self, bottom, top):
        """Computes the forward pass."""
        # Get features and output
        features = bottom[0].data()
        output = top[0].init_data(
            features.shape[:-1] + (self._num_output,), features.dtype,
            setdata=False)
        # initialize weights
        if not self._weight.has_data():
            self._weight.init_data(
                (features.shape[-1], self._num_output), features.dtype)
        if self._has_bias and not self._bias.has_data():
            self._bias.init_data((self._num_output), features.dtype)
        # computation
        weight = self._weight.data()
        blasdot.dot_lastdim(features, weight, out=output)
        if self._has_bias:
            output += self._bias.data()

    def backward(self, bottom, top, propagate_down):
        """Computes the backward pass."""
        # get diff
        top_diff = top[0].diff()
        features = bottom[0].data()
        # compute the gradient
        weight_diff = self._weight.init_diff(setzero=False)
        blasdot.dot_firstdims(features, top_diff, out=weight_diff)
        if self._has_bias:
            bias_diff = self._bias.init_diff(setzero=False)
            bias_diff[:] = top_diff.reshape(
                np.prod(top_diff.shape[:-1]), top_diff.shape[-1]).sum(0)
        # If necessary, compute the bottom Blob gradient.
        if propagate_down:
            bottom_diff = bottom[0].init_diff(setzero=False)
            blasdot.dot_lastdim(top_diff, self._weight.data().T,
                                out=bottom_diff)
        if self._reg is not None:
            return self._reg.reg(self._weight)
        else:
            return 0.

    def update(self):
        """Updates the parameters."""
        self._weight.update()
        if self._has_bias:
            self._bias.update()

