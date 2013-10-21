"""Implements the padding layer."""

from decaf import base

class PaddingLayer(base.Layer):
    """A layer that pads a matrix."""

    def __init__(self, **kwargs):
        """Initializes a padding layer.
        kwargs:
            'pad': the number of pixels to pad. Should be nonnegative.
                If pad is 0, the layer will simply mirror the input.
            'value': the value inserted to the padded area. Default 0.
        """
        base.Layer.__init__(self, **kwargs)
        self._pad = self.spec['pad']
        self._value = self.spec.get('value', 0)
        if self._pad < 0:
            raise ValueError('Padding should be nonnegative.')

    def forward(self, bottom, top):
        """Computes the forward pass."""
        if self._pad == 0:
            top[0].mirror(bottom[0])
            return
        # Get features and output
        features = bottom[0].data()
        if features.ndim != 4:
            raise ValueError('Bottom data should be a 4-dim tensor.')
        pad = self._pad
        newshape = (features.shape[0],
                    features.shape[1] + pad * 2,
                    features.shape[2] + pad * 2,
                    features.shape[3])
        output = top[0].init_data(newshape,
                                  features.dtype, setdata=False)
        output[:] = self._value
        output[:, pad:-pad, pad:-pad] = features

    def backward(self, bottom, top, propagate_down):
        """Computes the backward pass."""
        if not propagate_down:
            return 0.
        if self._pad == 0:
            bottom[0].mirror_diff(top[0].diff())
        else:
            pad = self._pad
            top_diff = top[0].diff()
            bottom_diff = bottom[0].init_diff(setzero=False)
            bottom_diff[:] = top_diff[:, pad:-pad, pad:-pad]
        return 0.

    def update(self):
        """Padding has nothing to update."""
        pass
