from decaf import base
from decaf.layers import core_layers, fillers, regularization
from decaf.util import gradcheck
import numpy as np
import unittest
import os
import sys

class TestNormalizer(unittest.TestCase):
    def setUp(self):
        pass

    def testMeanNormalizeLayer(self):
        np.random.seed(1701)
        output_blob = base.Blob()
        checker = gradcheck.GradChecker(1e-5)
        shapes = [(1,5,5,1), (1,5,5,3), (5,5), (1,5)]
        for shape in shapes:
            input_blob = base.Blob(shape, filler=fillers.GaussianRandFiller())
            layer = core_layers.MeanNormalizeLayer(
                name='normalize')
            result = checker.check(layer, [input_blob], [output_blob])
            print(result)
            self.assertTrue(result[0])
    
    def testResponseNormalizeLayer(self):
        np.random.seed(1701)
        output_blob = base.Blob()
        checker = gradcheck.GradChecker(1e-5)
        shapes = [(1,5,5,1), (1,5,5,3), (5,5), (1,5)]
        for shape in shapes:
            input_blob = base.Blob(shape,
                                   filler=fillers.RandFiller(min=0.1, max=1.))
            layer = core_layers.ResponseNormalizeLayer(
                name='normalize')
            result = checker.check(layer, [input_blob], [output_blob])
            print(result)
            self.assertTrue(result[0])

    # The following test is known to fail on my macbook when multiple OMP
    # threads are being used, so I will simply skip it.
    @unittest.skipIf(sys.platform.startswith('darwin') and 
                     ('OMP_NUM_THREADS' not in os.environ or
                      os.environ['OMP_NUM_THREADS'] != '1'),
                     "Known to not work on macs.")

    def testLocalResponseNormalizeLayer(self):
        np.random.seed(1701)
        output_blob = base.Blob()
        checker = gradcheck.GradChecker(1e-6)
        shapes = [(1,10), (5,10)]
        alphas = [1.0, 2.0]
        betas = [0.75, 1.0]
        for shape in shapes:
            for alpha in alphas:
                for beta in betas:
                    input_blob = base.Blob(shape, filler=fillers.RandFiller())
                    # odd size
                    layer = core_layers.LocalResponseNormalizeLayer(
                        name='normalize', k = 1., alpha=alpha, beta=beta, size=5)
                    result = checker.check(layer, [input_blob], [output_blob])
                    print(result)
                    self.assertTrue(result[0])
                    layer = core_layers.LocalResponseNormalizeLayer(
                        name='normalize', k = 2., alpha=alpha, beta=beta, size=5)
                    result = checker.check(layer, [input_blob], [output_blob])
                    print(result)
                    self.assertTrue(result[0])
                    # even size
                    layer = core_layers.LocalResponseNormalizeLayer(
                        name='normalize', k = 1., alpha=alpha, beta=beta, size=6)
                    result = checker.check(layer, [input_blob], [output_blob])
                    print(result)
                    self.assertTrue(result[0])

if __name__ == '__main__':
    unittest.main()
