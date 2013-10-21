"""Implements the convolution layer."""

from decaf import base
from decaf.layers import convolution

class GroupConvolutionLayer(base.Layer):
    """A layer that implements the block group convolution function."""

    def __init__(self, **kwargs):
        """Initializes the convolution layer. Strictly, this is a correlation
        layer since the kernels are not reversed spatially as in a classical
        convolution operation.

        kwargs:
            name: the name of the layer.
            group: the number of groups that should be carried out for the
                block group convolution. Note that the number of channels of
                the incoming block should be divisible by the number of groups,
                otherwise we will have an error produced.
            num_kernels: the number of kernels PER GROUP. As a result, the
                output would have (num_kernels * group) channels.
        Also the layer should be provided all the appropriate parameters for
        the underlying convolutional layer.
        """
        base.Layer.__init__(self, **kwargs)
        self._group = self.spec['group']
        self._conv_args = dict(self.spec)
        self._conv_args['name'] = self.spec['name'] + '_sub'
        del self._conv_args['group']
        self._bottom_sub = [base.Blob() for _ in range(self._group)]
        self._top_sub = [base.Blob() for _ in range(self._group)]
        self._conv_layers = None
        self._blocksize = 0
        self._num_kernels = self.spec['num_kernels']
        # create the convolution layers
        self._conv_layers = [
            convolution.ConvolutionLayer(**self._conv_args)
            for i in range(self._group)]
        self._param = sum((layer.param() for layer in self._conv_layers), [])
        return

    def forward(self, bottom, top):
        """Runs the forward pass."""
        bottom_data = bottom[0].data()
        if bottom_data.ndim != 4:
            raise ValueError('Bottom data should be a 4-dim tensor.')
        if bottom_data.shape[-1] % self._group:
            raise RuntimeError('The number of input channels (%d) should be'
                               ' divisible by the number of groups (%d).' %
                               (bottom_data.shape[-1], self._group))
        self._blocksize = bottom_data.shape[-1] / self._group
        for i in range(self._group):
            in_start = i * self._blocksize
            in_end = in_start + self._blocksize
            out_start = i * self._num_kernels
            out_end = out_start + self._num_kernels
            # Now, create intermediate blobs, and compute forward by group
            bottom_sub_data = self._bottom_sub[i].init_data(
                bottom_data.shape[:-1] + (self._blocksize,),
                bottom_data.dtype, setdata=False)
            bottom_sub_data[:] = bottom_data[:, :, :, in_start:in_end]
            self._conv_layers[i].forward([self._bottom_sub[i]],
                                         [self._top_sub[i]])
            top_sub_data = self._top_sub[i].data()
            if i == 0:
                top_data = top[0].init_data(
                    top_sub_data.shape[:-1] + \
                    (top_sub_data.shape[-1] * self._group,),
                    top_sub_data.dtype, setdata=False)
            top_data[:, :, :, out_start:out_end] = top_sub_data
        return

    def backward(self, bottom, top, propagate_down):
        """Runs the backward pass."""
        loss = 0.
        top_diff = top[0].diff()
        bottom_data = bottom[0].data()
        # initialize the sub diff
        if propagate_down:
            bottom_diff = bottom[0].init_diff(setzero=False)
        for i in range(self._group):
            top_sub_diff = self._top_sub[i].init_diff(setzero=False)
            bottom_sub_data = self._bottom_sub[i].data()
            in_start = i * self._blocksize
            in_end = in_start + self._blocksize
            out_start = i * self._num_kernels
            out_end = out_start + self._num_kernels
            # Since the convolutional layers will need the input data,
            # we will need to provide them.
            bottom_sub_data[:] = bottom_data[:, :, :, in_start:in_end]
            top_sub_diff[:] = top_diff[:, :, :, out_start:out_end]
            loss += self._conv_layers[i].backward(
                [self._bottom_sub[i]], [self._top_sub[i]], propagate_down)
            if propagate_down:
                bottom_sub_diff = self._bottom_sub[i].init_diff(setzero=False)
                bottom_diff[:, :, :, in_start:in_end] = bottom_sub_diff
        return loss

    def __getstate__(self):
        """When pickling, we will remove the intermediate data."""
        self._bottom_sub = [base.Blob() for _ in range(self._group)]
        self._top_sub = [base.Blob() for _ in range(self._group)]
        return self.__dict__
    
    def update(self):
        """updates the parameters."""
        for layer in self._conv_layers:
            layer.update()
