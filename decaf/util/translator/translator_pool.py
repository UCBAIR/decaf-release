"""Translates the pooling layers."""
from decaf.util.translator import registerer
from decaf.layers import core_layers
import math

def translator_pool(cuda_layer, output_shapes):
    """Translates the pooling layers."""
    method = cuda_layer['pool']
    if method == 'max':
        pass
    elif method == 'avg':
        # We have a slightly different name
        method = 'ave'
    else:
        raise NotImplementedError('Unrecognized pooling method: %s' % method)
    if cuda_layer['start'] != 0:
        raise NotImplementedError('Unsupported layer with a non-zero start.')
    # Check the outputsX size.
    output_size = math.ceil(
        float(cuda_layer['imgSize'] - cuda_layer['sizeX']) / 
        cuda_layer['stride']) + 1
    if cuda_layer['outputsX'] != output_size:
        raise NotImplementedError('Unsupported layer with custon output size.')
    # If all checks passed, we will return our pooling layer
    psize = cuda_layer['sizeX']
    stride = cuda_layer['stride']
    input_shape = output_shapes[cuda_layer['inputLayers'][0]['name']]
    output_shape = (
        int(math.ceil(float(input_shape[0] - psize) / stride)) + 1,
        int(math.ceil(float(input_shape[1] - psize) / stride)) + 1,
        input_shape[2])
    output_shapes[cuda_layer['name']] = output_shape
    return core_layers.PoolingLayer(
        name=cuda_layer['name'],
        psize=psize,
        stride=stride,
        mode=method)
    

registerer.register_translator('pool', translator_pool)
