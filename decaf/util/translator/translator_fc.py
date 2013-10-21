"""translator_fc translates a fully connected layer to a decaf
InnerProductLayer.
"""
from decaf.util.translator import registerer
from decaf.layers import core_layers
import numpy as np
from operator import mul

def translator_fc(cuda_layer, output_shapes):
    """The translator for the fc layer."""
    input_shape = output_shapes[cuda_layer['inputLayers'][0]['name']]
    input_size = reduce(mul, input_shape)
    num_output = cuda_layer['outputs']
    output_shapes[cuda_layer['name']] = (num_output,)
    decaf_layer = core_layers.InnerProductLayer(
        name=cuda_layer['name'],
        num_output=num_output)
    # put the parameters
    params = decaf_layer.param()
    # weight
    weight = cuda_layer['weights'][0]
    if weight.shape[0] != input_size or weight.shape[1] != num_output:
        raise ValueError('Incorrect shapes: weight shape %s, input shape %s,'
                         ' num_output %d' %
                         (weight.shape, input_shape, num_output))
    if len(input_shape) == 3:
        # The original input is an image, so we will need to reshape it
        weight = weight.reshape(
            (input_shape[2], input_shape[0], input_shape[1], num_output))
        converted_weight = np.empty(input_shape + (num_output,),
                                    weight.dtype)
        for i in range(input_shape[2]):
            converted_weight[:, :, i, :] = weight[i, :, :, :]
        converted_weight.resize(input_size, num_output)
    else:
        converted_weight = weight
    params[0].mirror(converted_weight)
    bias = cuda_layer['biases'][0]
    params[1].mirror(bias)
    if len(input_shape) == 1:
        return decaf_layer
    else:
        # If the input is not a vector, we need to have a flatten layer first.
        return [core_layers.FlattenLayer(name=cuda_layer['name'] + '_flatten'),
                decaf_layer]

registerer.register_translator('fc', translator_fc)
