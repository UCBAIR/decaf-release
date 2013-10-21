from decaf import base
from decaf.layers import convolution
import numpy as np
import time

from theano.tensor.nnet import conv
import theano
import theano.tensor as T

def theano_convolution(input_size, dtype, num_kernels, ksize, mode, iternum):
    rng = np.random.RandomState(23455)
    # instantiate 4D tensor for input
    if dtype == np.float32:
        input = T.tensor4(name='input', dtype='float32')
    else:
        input = T.tensor4(name='input', dtype='float64')
    # initialize shared variable for weights.
    w_shp = (num_kernels, input_size[-1], ksize, ksize)
    w_bound = np.sqrt(input_size[-1] * ksize * ksize)
    W = theano.shared( np.asarray(
                rng.uniform(
                    low=-1.0 / w_bound,
                    high=1.0 / w_bound,
                    size=w_shp),
                dtype=dtype), name ='W')
    conv_out = conv.conv2d(input, W, border_mode=mode)
    # create theano function to compute filtered images
    f = theano.function([input], conv_out)
    img = np.random.random_sample(input_size).astype(dtype)
    # put image in 4D tensor of shape (1, 3, height, width)
    img_ = img.swapaxes(0, 2).swapaxes(1, 2).reshape(1, input_size[-1], input_size[0], input_size[1])
    img_ = np.ascontiguousarray(img_)
    # just in case theano want to initialize something, we will run the function once first.
    filtered_img = f(img_)
    start = time.time()
    for i in range(iternum):
        filtered_img = f(img_)
    print 'theano time:', (time.time() - start) / iternum

def decaf_convolution(input_size, dtype, num_kernels, ksize, stride, mode, iternum):
    bottom = base.Blob((1,) + input_size, dtype=dtype)
    layer = convolution.ConvolutionLayer(
        name='conv', num_kernels=num_kernels, ksize=ksize,
        stride=stride, mode=mode)
    top = base.Blob()
    # run a forward pass first to initialize everything.
    layer.forward([bottom], [top])
    top.init_diff()
    top.diff().flat = 1.
    print '*****'
    print 'input shape:', bottom.data().shape[1:]
    start = time.time()
    for i in range(iternum):
        layer.forward([bottom], [top])
    print 'forward runtime:', (time.time() - start) / iternum
    print 'output shape:', top.data().shape[1:]
    start = time.time()
    for i in range(iternum):
        layer.backward([bottom], [top], True)
    print 'backward runtime:', (time.time() - start) / iternum
    print '*****'

if __name__ == '__main__':
    print 'test float32'
    theano_convolution((256,256,3), np.float32, 16, 11,    'full', 50)
    decaf_convolution((256,256,3), np.float32, 16, 11, 1, 'full', 50)
    theano_convolution((256,256,3), np.float32, 16, 11,    'valid', 50)
    decaf_convolution((256,256,3), np.float32, 16, 11, 1, 'valid', 50)
    print 'test thick convolution'
    theano_convolution((55,55,96), np.float32, 256, 3,    'full', 50)
    decaf_convolution((55,55,96), np.float32, 256, 3, 1, 'full', 50)
    theano_convolution((55,55,96), np.float32, 256, 3,    'valid', 50)
    decaf_convolution((55,55,96), np.float32, 256, 3, 1, 'valid', 50)

