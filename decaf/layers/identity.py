"""Implements a dummy identity layer."""

from decaf import base

class IdentityLayer(base.Layer):
    """A layer that does nothing but mirroring things."""

    def __init__(self, **kwargs):
        """Initializes an identity layer.

        kwargs:
            name: the layer name.
        """
        base.Layer.__init__(self, **kwargs)

    def forward(self, bottom, top):
        """Computes the forward pass."""
        for top_blob, bottom_blob in zip(top, bottom):
            top_blob.mirror(bottom_blob)

    def backward(self, bottom, top, propagate_down):
        """Computes the backward pass."""
        if not propagate_down:
            return 0.
        for top_blob, bottom_blob in zip(top, bottom):
            bottom_blob.mirror_diff(top_blob)
        return 0.

    def update(self):
        """Identity Layer has nothing to update."""
        pass
