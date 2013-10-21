from decaf import base
from decaf.layers import core_layers, fillers, regularization
from decaf.util import gradcheck
import numpy as np
import unittest


class TestDropoutGrad(unittest.TestCase):
    def setUp(self):
        pass

    def testDropoutGrad(self):
        np.random.seed(1701)
        input_blob = base.Blob((4,3), filler=fillers.GaussianRandFiller())
        output_blob = base.Blob()
        checker = gradcheck.GradChecker(1e-5)
        
        layer = core_layers.DropoutLayer(name='dropout', ratio=0.5,
                                         debug_freeze=True)
        result = checker.check(layer, [input_blob], [output_blob])
        print(result)
        self.assertTrue(result[0])

if __name__ == '__main__':
    unittest.main()
