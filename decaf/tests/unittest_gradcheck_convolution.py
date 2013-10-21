from decaf import base
from decaf.layers import core_layers, fillers, regularization
from decaf.util import gradcheck
import numpy as np
import unittest


class TestConvolutionGrad(unittest.TestCase):
    def setUp(self):
        pass

    def testConvolutionGrad(self):
        np.random.seed(1701)
        output_blob = base.Blob()
        checker = gradcheck.GradChecker(1e-4)
        shapes = [(1,5,5,1), (1,5,5,3)]
        num_kernels = 2
        params = [(3,1,'valid'), (3,1,'same'), (3,1,'full'), (2,1,'valid'), (2,1,'full'),
                  (3,2,'valid'), (3,2,'same'), (3,2,'full')]
        for shape in shapes:
            for ksize, stride, mode in params:
                print(ksize, stride, mode, shape)
                input_blob = base.Blob(shape, filler=fillers.GaussianRandFiller())
                layer = core_layers.ConvolutionLayer(
                    name='conv', ksize=ksize, stride=stride, mode=mode,
                    num_kernels=num_kernels,
                    filler=fillers.GaussianRandFiller())
                result = checker.check(layer, [input_blob], [output_blob])
                print(result)
                self.assertTrue(result[0])

if __name__ == '__main__':
    unittest.main()
