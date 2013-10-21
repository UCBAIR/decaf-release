from decaf import base
from decaf.layers import core_layers, fillers, regularization
from decaf.util import gradcheck
import numpy as np
import unittest


class TestPaddingGrad(unittest.TestCase):
    def setUp(self):
        pass

    def testPaddingGrad(self):
        np.random.seed(1701)
        output_blob = base.Blob()
        checker = gradcheck.GradChecker(1e-5)
        shapes = [(1,5,5,1), (1,5,5,3), (1,4,3,1), (1,4,3,3)]
        pads = [1,2,3]
        for pad in pads:
            for shape in shapes:
                input_blob = base.Blob(shape, filler=fillers.GaussianRandFiller())
                layer = core_layers.PaddingLayer(name='padding', pad=pad)
                result = checker.check(layer, [input_blob], [output_blob])
                print(result)
                self.assertTrue(result[0])

if __name__ == '__main__':
    unittest.main()
