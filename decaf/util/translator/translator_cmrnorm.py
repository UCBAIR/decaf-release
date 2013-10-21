"""Translates the cmrnorm layer."""
from decaf.util.translator import registerer
from decaf.layers import core_layers


def translator_cmrnorm(cuda_layer, output_shapes):
    """Translates the cmrnorm layer.
    Note: we hard-code the constant in the local response normalization
    layer to be 1. This may be different from Krizhevsky's NIPS paper but
    matches the actual cuda convnet code.
    """
    input_shape = output_shapes[cuda_layer['inputLayers'][0]['name']]
    output_shapes[cuda_layer['name']] = input_shape
    return core_layers.LocalResponseNormalizeLayer(
        name=cuda_layer['name'],
        size=cuda_layer['size'],
        k=1,
        alpha = cuda_layer['scale'] * cuda_layer['size'],
        beta = cuda_layer['pow'])

registerer.register_translator('cmrnorm', translator_cmrnorm)

