"""Implements the convolution layer."""

from decaf import base
from decaf.layers.cpp import wrapper
from decaf.util import blasdot
import numpy as np

# pylint: disable=R0902
class DeconvolutionLayer(base.Layer):
    """A layer that implements the deconvolution function: it is the inverse
    as the convolution operation.
    """

    def __init__(self, **kwargs):
        """Initializes the deconvolution layer. Strictly, this is a correlation
        layer since the kernels are not reversed spatially as in a classical
        convolution operation.

        kwargs:
            name: the name of the layer.
            num_channels: the number of output channels.
            ksize: the kernel size. Kernels will be square shaped and have the
                same number of channels as the data.
            stride: the kernel stride.
            mode: 'valid', 'same', or 'full'. The modes represent the corres-
                ponding convolution operation.
            reg: the regularizer to be used to add regularization terms.
                should be a decaf.base.Regularizer instance. Default None. 
            filler: a filler to initialize the weights. Should be a
                decaf.base.Filler instance. Default None.
        """
        base.Layer.__init__(self, **kwargs)
        self._num_channels = self.spec['num_channels']
        self._ksize = self.spec['ksize']
        self._stride = self.spec['stride']
        self._mode = self.spec['mode']
        self._reg = self.spec.get('reg', None)
        self._filler = self.spec.get('filler', None)
        self._memory = self.spec.get('memory', 1e7)
        if self._ksize <= 1:
            raise ValueError('Invalid kernel size. Kernel size should > 1.')
        if self._mode == 'same' and self._ksize % 2 == 0:
            raise ValueError('The "same" mode should have an odd kernel size.')
        # since the im2col operation often creates large intermediate matrices,
        # we will have intermediate blobs to store them.
        self._padded = base.Blob()
        self._col = base.Blob()
        # set up the parameter
        self._kernels = base.Blob(filler=self._filler)
        self._param = [self._kernels]
        # compute the border.
        if self._mode == 'valid':
            self._border = 0
        elif self._mode == 'same':
            self._border = self._ksize / 2
        elif self._mode == 'full':
            self._border = self._ksize - 1

    def forward(self, bottom, top):
        """Runs the forward pass."""
        bottom_data = bottom[0].data()
        if bottom_data.ndim != 4:
            raise ValueError('Bottom data should be a 4-dim tensor.')
        if not self._kernels.has_data():
            # initialize the kernels
            self._kernels.init_data(
                (bottom_data.shape[-1],
                 self._ksize * self._ksize * self._num_channels),
                bottom_data.dtype)
        # initialize the buffers.
        self._col.init_data((1, bottom_data.shape[1], bottom_data.shape[2],
                             self._kernels.data().shape[1]),
                            dtype = bottom_data.dtype)
        pad_height = self._ksize + (bottom_data.shape[1] - 1) \
                * self._stride
        pad_width = self._ksize + (bottom_data.shape[2] - 1) \
                * self._stride
        if self._mode != 'valid':
            padded_data = self._padded.init_data(
                (1, pad_height, pad_width, self._num_channels),
                dtype = bottom_data.dtype)
        top_data = top[0].init_data(
            (bottom_data.shape[0], pad_height - self._border * 2,
             pad_width - self._border * 2, self._num_channels),
            dtype=bottom_data.dtype)
        # process data individually
        for i in range(bottom_data.shape[0]):
            # first, compute the convolution as a gemm operation
            blasdot.dot_lastdim(bottom_data[i:i+1], self._kernels.data(),
                                out=self._col.data())
            if self._mode != 'valid':
            # do col2im
                wrapper.im2col_backward(padded_data, self._col.data(),
                               self._ksize, self._stride)
                top_data[i] = padded_data[0, self._border:-self._border,
                                          self._border:-self._border]
            else:
                wrapper.im2col_backward(top_data[i:i+1], self._col.data(),
                                        self._ksize, self._stride)
        return

    def backward(self, bottom, top, propagate_down):
        """Runs the backward pass."""
        top_diff = top[0].diff()
        bottom_data = bottom[0].data()
        kernel_diff = self._kernels.init_diff()
        kernel_diff_buffer = np.zeros_like(kernel_diff)
        col_diff = self._col.init_diff()
        if propagate_down:
            bottom_diff = bottom[0].init_diff()
        if self._mode != 'valid':
            pad_diff = self._padded.init_diff()
        for i in range(bottom_data.shape[0]):
            if self._mode != 'valid':
                # do padding
                pad_diff[0, self._border:-self._border,
                         self._border:-self._border] = top_diff[i]
            else:
                pad_diff = top_diff[i:i+1].view()
            # run im2col
            wrapper.im2col_forward(pad_diff, col_diff, self._ksize,
                                   self._stride)
            blasdot.dot_firstdims(bottom_data[i], col_diff,
                                 out=kernel_diff_buffer)
            kernel_diff += kernel_diff_buffer
            if propagate_down:
                # compute final gradient
                blasdot.dot_lastdim(col_diff, self._kernels.data().T,
                                    out=bottom_diff[i])
        # finally, add the regularization term
        if self._reg is not None:
            return self._reg.reg(self._kernels, bottom_data.shape[0])
        else:
            return 0.

    def __getstate__(self):
        """When pickling, we will remove the intermediate data."""
        self._padded = base.Blob()
        self._col = base.Blob()
        return self.__dict__

    def update(self):
        """updates the parameters."""
        # Only the inner product layer needs to be updated.
        self._kernels.update()

