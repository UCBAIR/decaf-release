"""Translates the softmax layers."""
from decaf.util.translator import registerer
from decaf.layers import core_layers


def translator_softmax(cuda_layer, output_shapes):
    """Translates the softmax layers."""
    input_shape = output_shapes[cuda_layer['inputLayers'][0]['name']]
    output_shapes[cuda_layer['name']] = input_shape
    return core_layers.SoftmaxLayer(
        name=cuda_layer['name'])

registerer.register_translator('softmax', translator_softmax)
