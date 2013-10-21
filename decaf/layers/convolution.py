"""Implements the convolution layer."""

from decaf import base
from decaf.layers.cpp import wrapper
from decaf.util import blasdot
import numpy as np

class ConvolutionLayer(base.Layer):
    """A layer that implements the convolution function."""

    def __init__(self, **kwargs):
        """Initializes the convolution layer. Strictly, this is a correlation
        layer since the kernels are not reversed spatially as in a classical
        convolution operation.

        kwargs:
            name: the name of the layer.
            num_kernels: the number of kernels.
            ksize: the kernel size. Kernels will be square shaped and have the
                same number of channels as the data.
            stride: the kernel stride.
            mode: 'valid', 'same', or 'full'.
            pad: if set, this value will overwrite the mode and we will use 
                the given pad size. Default None.
            reg: the regularizer to be used to add regularization terms.
                should be a decaf.base.Regularizer instance. Default None. 
            filler: a filler to initialize the weights. Should be a
                decaf.base.Filler instance. Default None.
            has_bias: specifying if the convolutional network should have a
                bias term. Note that the same bias is going to be applied
                regardless of the location. Default True.
            bias_filler: a filler to unitialize the bias. Should be a
                decaf.base.Filler instance. Default None.
            large_mem: if set True, the layer will consume a lot of memory by
                storing all the intermediate im2col results, but will increase
                the backward operation time. Default False.
        When computing convolutions, we will always start from the top left
        corner, and any rows/columns on the right and bottom sides that do not
        fit the stride will be discarded. To enforce the 'same' mode to return
        results of the same size as the data, we require the 'same' mode to be
        paired with an odd number as the kernel size.
        """
        base.Layer.__init__(self, **kwargs)
        self._num_kernels = self.spec['num_kernels']
        self._ksize = self.spec['ksize']
        self._stride = self.spec['stride']
        self._large_mem = self.spec.get('large_mem', False)
        self._reg = self.spec.get('reg', None)
        self._has_bias = self.spec.get('has_bias', True)
        if self._ksize <= 1:
            raise ValueError('Invalid kernel size. Kernel size should > 1.')
        # since the im2col operation often creates large intermediate matrices,
        # we will process them in batches.
        self._padded = base.Blob()
        self._col = base.Blob()
        # set up the parameter
        self._kernels = base.Blob(filler=self.spec.get('filler', None))
        if self._has_bias:
            self._bias = base.Blob(filler=self.spec.get('bias_filler', None))
            self._param = [self._kernels, self._bias]
        else:
            self._param = [self._kernels]
        self._pad_size = self.spec.get('pad', None)
        if self._pad_size is None:
            self._mode = self.spec['mode']
            if self._mode == 'same' and self._ksize % 2 == 0:
                raise ValueError(
                    'The "same" mode should have an odd kernel size.')
            if self._mode == 'valid':
                self._pad_size = 0
            elif self._mode == 'full':
                self._pad_size = self._ksize - 1
            elif self._mode == 'same':
                self._pad_size = int(self._ksize / 2)
            else:
                raise ValueError('Unknown mode: %s' % self._mode)
    
    def forward(self, bottom, top):
        """Runs the forward pass."""
        bottom_data = bottom[0].data()
        if bottom_data.ndim != 4:
            raise ValueError('Bottom data should be a 4-dim tensor.')
        if not self._kernels.has_data():
            # initialize the kernels
            self._kernels.init_data(
                (self._ksize * self._ksize * bottom_data.shape[-1],
                 self._num_kernels),
                bottom_data.dtype)
            if self._has_bias:
                self._bias.init_data((self._num_kernels,), bottom_data.dtype)
        # pad the data
        if self._pad_size == 0:
            padded_data = self._padded.mirror(bottom_data)
        else:
            padded_data = self._padded.init_data(
                    (bottom_data.shape[0],
                     bottom_data.shape[1] + self._pad_size * 2,
                     bottom_data.shape[2] + self._pad_size * 2,
                     bottom_data.shape[3]),
                    bottom_data.dtype)
            padded_data[:, self._pad_size:-self._pad_size,
                        self._pad_size:-self._pad_size] = bottom_data
        # initialize self._col
        if self._large_mem:
            col_data_num = bottom_data.shape[0]
        else:
            col_data_num = 1
        col_data = self._col.init_data(
            (col_data_num,
             (padded_data.shape[1] - self._ksize) / self._stride + 1,
             (padded_data.shape[2] - self._ksize) / self._stride + 1,
             padded_data.shape[3] * self._ksize * self._ksize),
            padded_data.dtype, setdata=False)
        # initialize top data
        top_data = top[0].init_data(
            (bottom_data.shape[0], col_data.shape[1], col_data.shape[2],
             self._num_kernels), dtype=bottom_data.dtype, setdata=False)
        # process data individually
        if self._large_mem:
            wrapper.im2col_forward(padded_data, col_data,
                                   self._ksize, self._stride)
            blasdot.dot_lastdim(col_data, self._kernels.data(), out=top_data)
        else:
            for i in range(bottom_data.shape[0]):
                # call im2col individually
                wrapper.im2col_forward(padded_data[i:i+1], col_data,
                                       self._ksize, self._stride)
                blasdot.dot_lastdim(col_data, self._kernels.data(),
                                    out=top_data[i])
        if self._has_bias:
            top_data += self._bias.data()
        return

    def backward(self, bottom, top, propagate_down):
        """Runs the backward pass."""
        top_diff = top[0].diff()
        padded_data = self._padded.data()
        col_data = self._col.data()
        bottom_data = bottom[0].data()
        if bottom_data.ndim != 4:
            raise ValueError('Bottom data should be a 4-dim tensor.')
        kernel_diff = self._kernels.init_diff()
        if self._has_bias:
            bias_diff = self._bias.init_diff()
            # bias diff is fairly easy to compute: just sum over all other
            # dimensions
            np.sum(top_diff.reshape(top_diff.size / top_diff.shape[-1],
                                    top_diff.shape[-1]),
                   axis=0, out=bias_diff)
        if propagate_down:
            bottom_diff = bottom[0].init_diff(setzero=False)
            col_diff = self._col.init_diff()
            if self._pad_size == 0:
                padded_diff = self._padded.mirror_diff(bottom_diff)
            else:
                padded_diff = self._padded.init_diff(setzero=False)
        if self._large_mem:
            # we have the col_data all pre-stored, making things more efficient.
            blasdot.dot_firstdims(col_data, top_diff, out=kernel_diff)
            if propagate_down:
                blasdot.dot_lastdim(top_diff, self._kernels.data().T,
                                    out=col_diff)
                wrapper.im2col_backward(padded_diff, col_diff,
                                    self._ksize, self._stride)
        else:
            kernel_diff_buffer = np.zeros_like(kernel_diff)
            for i in range(bottom_data.shape[0]):
                # although it is a backward layer, we still need to compute
                # the intermediate results using forward calls.
                wrapper.im2col_forward(padded_data[i:i+1], col_data,
                                       self._ksize, self._stride)
                blasdot.dot_firstdims(col_data, top_diff[i],
                                     out=kernel_diff_buffer)
                kernel_diff += kernel_diff_buffer
                if propagate_down:
                    blasdot.dot_lastdim(top_diff[i], self._kernels.data().T,
                                        out=col_diff)
                    # im2col backward
                    wrapper.im2col_backward(padded_diff[i:i+1], col_diff,
                                            self._ksize, self._stride)
        # finally, copy results to the bottom diff.
        if propagate_down:
            if self._pad_size != 0:
                bottom_diff[:] = padded_diff[:,
                                             self._pad_size:-self._pad_size,
                                             self._pad_size:-self._pad_size]
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
        if self._has_bias:
            self._bias.update()

