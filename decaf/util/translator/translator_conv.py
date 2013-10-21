"""Translates the convolution and group convolution layers."""
from decaf.util.translator import registerer
from decaf.layers import core_layers
import numpy as np

#pylint: disable=R0914
def translator_conv(cuda_layer, output_shapes):
    """Translates the convolution and group convolution layers."""
    group = cuda_layer['groups'][0]
    num_kernels = cuda_layer['filters']
    if num_kernels % group:
        raise ValueError('Incorrect num_kernels and group combination.')
    ksize = cuda_layer['filterSize'][0]
    if not cuda_layer['sharedBiases']:
        raise ValueError('Unshared bias layers not supported yet.')
    stride = cuda_layer['stride'][0]
    pad = -cuda_layer['padding'][0]
    # figure out the output shape
    input_shape = output_shapes[cuda_layer['inputLayers'][0]['name']]
    padded_shape = (input_shape[0] + pad * 2,
                    input_shape[1] + pad * 2,
                    input_shape[2])
    output_shape = ((padded_shape[0] - ksize) / stride + 1,
                    (padded_shape[1] - ksize) / stride + 1,
                    num_kernels)
    output_shapes[cuda_layer['name']] = output_shape
    weight = cuda_layer['weights'][0]
    input_channels = cuda_layer['channels'][0] / group
    weight.resize((input_channels, ksize, ksize, num_kernels))
    converted_weight = np.empty((ksize, ksize, input_channels, num_kernels),
                                weight.dtype)
    for i in range(input_channels):
        converted_weight[:, :, i, :] = weight[i, :, :, :]
    converted_weight.resize(ksize * ksize * input_channels, num_kernels)

    bias = cuda_layer['biases'].flatten().copy()
    if group == 1:
        # We should return a simple convolution layer
        decaf_layer = core_layers.ConvolutionLayer(
            name=cuda_layer['name'],
            num_kernels=num_kernels,
            ksize=ksize,
            stride=stride,
            pad=pad)
        param = decaf_layer.param()
        param[0].mirror(converted_weight)
        param[1].mirror(bias)
    else:
        # We should return a grouped convolution layer
        num_divided_kernels = num_kernels / group
        decaf_layer = core_layers.GroupConvolutionLayer(
            name=cuda_layer['name'],
            num_kernels=num_divided_kernels,
            ksize=ksize,
            stride=stride,
            pad=pad,
            group=group)
        param = decaf_layer.param()
        curr = 0
        for i in range(0, group * 2, 2):
            param[i].mirror(
                converted_weight[:, curr:curr+num_divided_kernels].copy())
            param[i+1].mirror(bias[curr:curr+num_divided_kernels])
            curr += num_divided_kernels
    return decaf_layer

registerer.register_translator('conv', translator_conv)
