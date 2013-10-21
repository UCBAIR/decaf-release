from decaf import base
from decaf.layers import core_layers, fillers, regularization
import numpy as np
import unittest


class TestIm2col(unittest.TestCase):
    def setUp(self):
        pass

    def testIm2col(self):
        np.random.seed(1701)
        output_blob = base.Blob()
        shapes = [(1,5,5,1), (1,5,5,3), (1,4,3,1), (1,4,3,3),
                  (3,5,5,1), (3,5,5,3), (3,4,3,1), (3,4,3,3)]
        params = [(2,1), (2,2), (3,1), (3,2)] 
        for psize, stride in params:
            for shape in shapes:
                input_blob = base.Blob(shape, filler=fillers.GaussianRandFiller())
                layer = core_layers.Im2colLayer(name='im2col',
                                                psize=psize, stride=stride)
                layer.forward([input_blob], [output_blob])
                # compare against naive implementation
                for i in range(0, shape[1] - psize - 1, stride):
                    for j in range(0, shape[2] - psize - 1, stride):
                        np.testing.assert_array_almost_equal(
                            output_blob.data()[:, i, j].flatten(),
                            input_blob.data()[:,
                                              i*stride:i*stride+psize,
                                              j*stride:j*stride+psize,
                                              :].flatten())

if __name__ == '__main__':
    unittest.main()
