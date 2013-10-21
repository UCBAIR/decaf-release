from decaf import base
from decaf.layers import core_layers, fillers
from decaf.util import gradcheck
import numpy as np
import unittest


class TestSoftmaxGrad(unittest.TestCase):
    def setUp(self):
        pass

    def testSoftmaxGrad(self):
        np.random.seed(1701)
        input_blob = base.Blob((10,5), filler=fillers.GaussianRandFiller())
        output_blob = base.Blob()
        layer = core_layers.SoftmaxLayer(name='softmax')
        checker = gradcheck.GradChecker(1e-5)
        result = checker.check(layer, [input_blob], [output_blob])
        print(result)
        self.assertTrue(result[0])
        # Also, let's check the result
        pred = input_blob.data()
        prob = np.exp(pred) / np.exp(pred).sum(1)[:, np.newaxis]
        np.testing.assert_array_almost_equal(
            output_blob.data(), prob)


if __name__ == '__main__':
    unittest.main()
