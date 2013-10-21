from decaf import base
from decaf.layers import core_layers, fillers, regularization
from decaf.util import gradcheck
import numpy as np
import unittest


class TestIm2colGrad(unittest.TestCase):
    def setUp(self):
        pass

    def testIm2colGrad(self):
        np.random.seed(1701)
        output_blob = base.Blob()
        checker = gradcheck.GradChecker(1e-4)
        shapes = [(1,5,5,1), (1,5,5,3), (1,4,3,1), (1,4,3,3)]
        params = [(2,1), (2,2), (3,1), (3,2)] 
        for psize, stride in params:
            for shape in shapes:
                input_blob = base.Blob(shape, filler=fillers.GaussianRandFiller())
                layer = core_layers.Im2colLayer(name='im2col', psize=psize, stride=stride)
                result = checker.check(layer, [input_blob], [output_blob])
                print(result)
                self.assertTrue(result[0])

if __name__ == '__main__':
    unittest.main()
