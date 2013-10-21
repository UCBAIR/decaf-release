"""Implements the minibatch sampling layer."""

from decaf import base
import numpy as np

class BasicMinibatchLayer(base.DataLayer):
    """A layer that extracts minibatches from bottom blobs.
    
    We will not randomly generate minibatches, but will instead produce them
    sequentially. Every forward() call will change the minibatch, so if you
    want a fixed minibatch, do NOT run forward multiple times.
    """

    def __init__(self, **kwargs):
        """Initializes the layer.

        kwargs:
            minibatch: the minibatch size.
        """
        base.DataLayer.__init__(self, **kwargs)
        self._minibatch = self.spec['minibatch']
        self._index = 0

    def forward(self, bottom, top):
        """Computes the forward pass."""
        size = bottom[0].data().shape[0]
        end_id = self._index + self._minibatch
        for bottom_blob, top_blob in zip(bottom, top):
            bottom_data = bottom_blob.data()
            if bottom_data.shape[0] != size:
                raise RuntimeError(
                    'Inputs do not have identical number of data points!')
            top_data = top_blob.init_data(
                (self._minibatch,) + bottom_data.shape[1:], bottom_data.dtype)
            # copy data
            if end_id <= size:
                top_data[:] = bottom_data[self._index:end_id]
            else:
                top_data[:(size - self._index)] = bottom_data[self._index:]
                top_data[-(end_id - size):] = bottom_data[:(end_id - size)]
        # finally, compute the new index.
        self._index = end_id % size
        

class RandomPatchLayer(base.DataLayer):
    """A layer that randomly extracts patches from bottom blobs.
    """

    def __init__(self, **kwargs):
        """Initialize the layer.

        kwargs:
            psize: the patch size.
            factor: the number of patches per bottom layer's image.
        """
        base.DataLayer.__init__(self, **kwargs)

    def forward(self, bottom, top):
        """Computes the forward pass."""
        factor = self.spec['factor']
        psize = self.spec['psize']
        bottom_data = bottom[0].data()
        num_img, height, width, num_channels = bottom_data.shape
        top_data = top[0].init_data(
            (num_img * factor, psize, psize, num_channels),
            dtype=bottom_data.dtype)
        h_indices = np.random.randint(height - psize, size=num_img * factor)
        w_indices = np.random.randint(width - psize, size=num_img * factor)
        for i in range(num_img):
            for j in range(factor):
                current = i * factor + j
                h_index, w_index = h_indices[current], w_indices[current]
                top_data[current] = bottom_data[i,
                                                h_index:h_index + psize,
                                                w_index:w_index + psize]
        return

