from decaf import base
from decaf.layers import core_layers, fillers, regularization
from decaf.util import gradcheck
import numpy as np
import unittest


class TestPoolingGrad(unittest.TestCase):
    def setUp(self):
        pass

    def testPoolingGrad(self):
        np.random.seed(1701)
        output_blob = base.Blob()
        checker = gradcheck.GradChecker(1e-4)
        shapes = [(1,7,7,1), (2,7,7,1), (1,7,7,3), (1,8,8,3), (1,13,13,1), (1,13,13,2)]
        params = [(3,2,'max'), (3,2,'ave'),(3,3,'max'), (3,3,'ave'),
                  (5,3,'max'), (5,3,'ave'),(5,5,'max'), (5,5,'ave')]
        for shape in shapes:
            for psize, stride, mode in params:
                print(psize, stride, mode, shape)
                input_blob = base.Blob(shape, filler=fillers.GaussianRandFiller())
                layer = core_layers.PoolingLayer(
                    name='pool', psize=psize, stride=stride, mode=mode)
                result = checker.check(layer, [input_blob], [output_blob])
                print(result)
                self.assertTrue(result[0])

if __name__ == '__main__':
    unittest.main()
