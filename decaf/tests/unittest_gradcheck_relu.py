from decaf import base
from decaf.layers import core_layers, fillers, regularization
from decaf.util import gradcheck
import numpy as np
import unittest


class TestReLUGrad(unittest.TestCase):
    def setUp(self):
        pass

    def testReLUGrad(self):
        np.random.seed(1701)
        shapes = [(4,3), (1,10), (2,5,5,1), (2,5,5,3)]
        output_blob = base.Blob()
        layer = core_layers.ReLULayer(name='relu')
        checker = gradcheck.GradChecker(1e-5)
        for shape in shapes:
            input_blob = base.Blob(shape, filler=fillers.GaussianRandFiller())
            result = checker.check(layer, [input_blob], [output_blob])
            print(result)
            self.assertTrue(result[0])

if __name__ == '__main__':
    unittest.main()
