from decaf import base
from decaf.layers import core_layers, fillers, regularization
from decaf.util import gradcheck
import numpy as np
import unittest


class TestSigmoidGrad(unittest.TestCase):
    def setUp(self):
        pass

    def testSigmoidGrad(self):
        np.random.seed(1701)
        shapes = [(4,3), (1,10), (2,5,5,1), (2,5,5,3)]
        output_blob = base.Blob()
        layer = core_layers.SigmoidLayer(name='sigmoid')
        checker = gradcheck.GradChecker(1e-5)
        for shape in shapes:
            input_blob = base.Blob(shape, filler=fillers.GaussianRandFiller())
            # let's check the forward results by the way
            layer.forward([input_blob], [output_blob])
            np.testing.assert_array_almost_equal(
                output_blob.data(), 1. / (1. + np.exp(-input_blob.data())))
            result = checker.check(layer, [input_blob], [output_blob])
            print(result)
            self.assertTrue(result[0])

if __name__ == '__main__':
    unittest.main()
