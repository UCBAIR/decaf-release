from decaf import base
from decaf.layers import core_layers, fillers, regularization
from decaf.util import gradcheck
import numpy as np
import unittest


class TestInnerproductGrad(unittest.TestCase):
    def setUp(self):
        pass

    def testInnerproductGrad(self):
        np.random.seed(1701)
        input_blob = base.Blob((4,3), filler=fillers.GaussianRandFiller())
        output_blob = base.Blob()
        checker = gradcheck.GradChecker(1e-5)
        
        ip_layer = core_layers.InnerProductLayer(
            name='ip', num_output=5, bias=True,
            filler=fillers.GaussianRandFiller(),
            bias_filler=fillers.GaussianRandFiller(),
            reg=None)
        result = checker.check(ip_layer, [input_blob], [output_blob])
        print(result)
        self.assertTrue(result[0])
        
        ip_layer = core_layers.InnerProductLayer(
            name='ip', num_output=5, bias=False,
            filler=fillers.GaussianRandFiller(),
            reg=None)
        result = checker.check(ip_layer, [input_blob], [output_blob])
        print(result)
        self.assertTrue(result[0])

        ip_layer = core_layers.InnerProductLayer(
            name='ip', num_output=5, bias=True,
            filler=fillers.GaussianRandFiller(),
            bias_filler=fillers.GaussianRandFiller(),
            reg=regularization.L2Regularizer(weight=0.1))
        result = checker.check(ip_layer, [input_blob], [output_blob])
        print(result)
        self.assertTrue(result[0])

if __name__ == '__main__':
    unittest.main()
