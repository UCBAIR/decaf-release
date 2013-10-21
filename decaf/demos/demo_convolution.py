"""This demo will show how we do simple convolution on the lena image with
a 15*15 average filter.
"""

from decaf import base
from decaf.util import smalldata
from decaf.layers import convolution, fillers
import numpy as np
from skimage import io

"""The main demo code."""
img = np.asarray(smalldata.lena())
img = img.reshape((1,) + img.shape).astype(np.float64)
# wrap the img in a blob
input_blob = base.Blob()
input_blob.mirror(img)

# create a convolutional layer
layer = convolution.ConvolutionLayer(
    name='convolution',
    num_kernels=1,
    ksize=15,
    stride=1,
    mode='same',
    filler=fillers.ConstantFiller(value=1./15/15/3))

# run the layer
output_blob = base.Blob()
layer.forward([input_blob], [output_blob])

out = output_blob.data()[0, :, :, 0].astype(np.uint8)
io.imsave('out.png', out)
print('Convolution result written to out.png')
