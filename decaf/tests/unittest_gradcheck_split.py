from decaf import base
from decaf.layers import fillers
from decaf.util import gradcheck
import numpy as np
import unittest


class TestSplitGrad(unittest.TestCase):
    def setUp(self):
        pass

    def testSplitGrad(self):
        np.random.seed(1701)
        output_blobs = [base.Blob(), base.Blob()]
        checker = gradcheck.GradChecker(1e-5)
        shapes = [(5,4), (5,1), (1,5), (1,5,5), (1,5,5,3), (1,5,5,1)]
        for shape in shapes:
            input_blob = base.Blob(shape, filler=fillers.GaussianRandFiller())
            layer = base.SplitLayer(name='split')
            result = checker.check(layer, [input_blob], output_blobs)
            print(result)
            self.assertTrue(result[0])

if __name__ == '__main__':
    unittest.main()
