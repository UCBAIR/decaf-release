"""Implements the im2col layer."""

from decaf import base
from decaf.layers.cpp import wrapper

class Im2colLayer(base.Layer):
    """A layer that implements the im2col function."""

    def __init__(self, **kwargs):
        """Initializes an im2col layer.

        kwargs:
            name: the name of the layer.
            psize: the patch size (patch will be a square).
            stride: the patch stride.

        If the input image has shape [height, width, nchannels], the output
        will have shape [(height-psize)/stride+1, (width-psize)/stride+1,
        nchannels * psize * psize].
        """
        base.Layer.__init__(self, **kwargs)
        self._psize = self.spec['psize']
        self._stride = self.spec['stride']
        if self._psize <= 1:
            raise ValueError('Padding should be larger than 1.')
        if self._stride < 1:
            raise ValueError('Stride should be larger than 0.')

    def _get_new_shape(self, features):
        """Gets the new shape of the im2col operation."""
        if features.ndim != 4:
            raise ValueError('Input features should be 4-dimensional.')
        num, height, width, channels = features.shape
        return (num,
                (height - self._psize) / self._stride + 1,
                (width - self._psize) / self._stride + 1,
                channels * self._psize * self._psize)

    def forward(self, bottom, top):
        """Computes the forward pass."""
        # Get features and output
        features = bottom[0].data()
        output = top[0].init_data(self._get_new_shape(features),
                                  features.dtype, setdata=False)
        wrapper.im2col_forward(features, output, self._psize, self._stride)

    def backward(self, bottom, top, propagate_down):
        """Computes the backward pass."""
        if not propagate_down:
            return 0.
        top_diff = top[0].diff()
        bottom_diff = bottom[0].init_diff(setzero=False)
        wrapper.im2col_backward(bottom_diff, top_diff, self._psize,
                                self._stride)
        return 0.

    def update(self):
        """Im2col has nothing to update."""
        pass
