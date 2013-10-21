"""Implements the Mean and variance normalization layer."""

from decaf import base
from decaf.layers.cpp import wrapper
import numpy as np
from numpy.core.umath_tests import inner1d


class MeanNormalizeLayer(base.Layer):
    """ A Layer that removes the mean along the mast dimension.
    """
    def __init__(self, **kwargs):
        base.Layer.__init__(self, **kwargs)

    def forward(self, bottom, top):
        """Computes the backward pass."""
        features = bottom[0].data()
        output = top[0].init_data(features.shape, features.dtype, setdata=False)
        # Create 2-dimenisonal views of the features and outputs.
        features.shape = (features.size / features.shape[-1], features.shape[-1])
        output.shape = features.shape
        output[:] = features
        output -= features.mean(axis=1)[:, np.newaxis]

    def backward(self, bottom, top, propagate_down):
        """Computes the backward pass."""
        if propagate_down:
            top_diff = top[0].diff()
            bottom_diff = bottom[0].init_diff(setzero=False)
            top_diff.shape = (top_diff.size / top_diff.shape[-1], top_diff.shape[-1])
            bottom_diff.shape = top_diff.shape
            bottom_diff[:] = top_diff
            bottom_diff -= (top_diff.sum(1) / top_diff.shape[-1])[:, np.newaxis]
        return 0.
        
    def update(self):
        """Has nothing to update."""
        pass


class ResponseNormalizeLayer(base.Layer):
    """A layer that normalizes the last dimension. For a vector x, it is 
    normalized as
        y_i = x_i / sqrt(smooth + 1/N \sum_j x_j^2),
    where N is the length of the vector.

    If you would like to subtract the mean and then normalize by standard
    deviation, stack a mean and response normalize layer.
    """
    def __init__(self, **kwargs):
        """Initalizes the layer. 
        
        kwargs:
            smooth: the smoothness term added to the norm.
        """
        base.Layer.__init__(self, **kwargs)
        self._scale = None

    def forward(self, bottom, top):
        """Computes the forward pass."""
        # Get features and output
        features = bottom[0].data()
        output = top[0].init_data(features.shape, features.dtype, setdata=False)
        # Create 2-dimenisonal views of the features and outputs.
        features.shape = (features.size / features.shape[-1], features.shape[-1])
        output.shape = features.shape
        self._scale = inner1d(features, features)
        self._scale /= features.shape[-1]
        self._scale += self.spec.get('smooth', np.finfo(self._scale.dtype).eps)
        np.sqrt(self._scale, out=self._scale)
        output[:] = features
        output /= self._scale[:, np.newaxis]
    
    def backward(self, bottom, top, propagate_down):
        """Computes the backward pass."""
        if propagate_down:
            features = bottom[0].data()
            output = top[0].data()
            top_diff = top[0].diff()
            bottom_diff = bottom[0].init_diff(setzero=False)
            scale = self._scale
            # Create 2-dimenisonal views of the features and outputs.
            features.shape = (features.size / features.shape[-1],
                              features.shape[-1])
            output.shape = features.shape
            top_diff.shape = features.shape
            bottom_diff.shape = features.shape
            # Compute gradients
            # TODO(Yangqing) any more efficient representations?
            bottom_diff[:] = top_diff / scale[:, np.newaxis] - output * \
                    (inner1d(top_diff, output) / scale / features.shape[-1])\
                    [:, np.newaxis]
        return 0.

    def update(self):
        """Has nothing to update."""
        pass


class LocalResponseNormalizeLayer(base.Layer):
    """A layer that locally normalizes the last dimension. For a vector x, it is 
    normalized as
        y_i = x_i / (k + alpha/size \sum_j x_j^2)^beta,
    where the range of j is
        [max(0, i - floor((size-1)/2)), min(dim, i - floor((size-1)/2) + size)].
    """
    def __init__(self, **kwargs):
        """Initalizes the layer. 
        
        kwargs:

            k, alpha, beta: as defined in the equation.
            size: the local range.
        """
        base.Layer.__init__(self, **kwargs)
        self._k = self.spec['k']
        self._alpha = self.spec['alpha']
        self._beta = self.spec['beta']
        self._size = self.spec['size']
        self._scale = base.Blob()

    def forward(self, bottom, top):
        """Computes the forward pass."""
        features = bottom[0].data()
        output = top[0].init_data(features.shape, features.dtype)
        scale = self._scale.init_data(features.shape, features.dtype)
        if self._size > features.shape[-1]:
            raise base.DecafError('Incorrect size: should be smaller than '
                                  'the number of input channels.')
        wrapper.lrn_forward(features, output, scale,
                            self._size, self._k, self._alpha, self._beta)
    
    def backward(self, bottom, top, propagate_down):
        """Computes the backward pass."""
        if propagate_down:
            features = bottom[0].data()
            output = top[0].data()
            top_diff = top[0].diff()
            bottom_diff = bottom[0].init_diff()
            scale = self._scale.data()
            wrapper.lrn_backward(
                features, output, bottom_diff, top_diff, scale,
                self._size, self._k, self._alpha, self._beta)
        return 0.

    def __getstate__(self):
        """When pickling, we will remove the intermediate data."""
        self._scale = base.Blob()
        return self.__dict__
    
    def update(self):
        """Has nothing to update."""
        pass

#    def _backward_python_implementation(self, bottom, top, propagate_down):
#        # This is a sample python implementation of the backward operation.
#        features = bottom[0].data()
#        output = top[0].data()
#        top_diff = top[0].diff()
#        bottom_diff = bottom[0].init_diff()
#        scale = self._scale.data()
#        features.shape = (features.size / features.shape[-1],
#                          features.shape[-1])
#        output.shape = top_diff.shape = bottom_diff.shape = scale.shape \
#                = features.shape
#        # elementwise gradient computation
#        for n in range(features.shape[0]):
#            for c in range(features.shape[1]):
#                local_end = c + (self._size + 1) / 2
#                local_start = local_end - self._size
#                local_start = max(local_start, 0)
#                local_end = min(local_end, features.shape[1])
#                bottom_diff[n, c] = \
#                    top_diff[n, c] / (scale[n, c] ** self._beta) - \
#                    2. * self._alpha * self._beta / self._size * \
#                    features[n, c] * \
#                    (output[n, local_start:local_end] * \
#                     top_diff[n, local_start:local_end] / \
#                     scale[n, local_start:local_end]).sum()
#        return 0

