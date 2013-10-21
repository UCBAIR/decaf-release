from decaf import base
from decaf.layers import core_layers, fillers, regularization
from decaf.util import gradcheck
import numpy as np
import unittest


class TestGroupConvolutionGrad(unittest.TestCase):
    def setUp(self):
        pass

    def testGroupConvolutionGrad(self):
        np.random.seed(1701)
        output_blob = base.Blob()
        checker = gradcheck.GradChecker(1e-3)
        shapes = [(1,5,5,4)]
        num_kernels = 1
        group = 2
        params = [(3,1,'valid'), (3,1,'same'), (3,1,'full'), (2,1,'valid'), (2,1,'full'),
                  (3,2,'valid'), (3,2,'same'), (3,2,'full')]
        for shape in shapes:
            for ksize, stride, mode in params:
                print(ksize, stride, mode, shape)
                input_blob = base.Blob(shape, filler=fillers.GaussianRandFiller())
                layer = core_layers.GroupConvolutionLayer(
                    name='gconv', ksize=ksize, stride=stride, mode=mode,
                    num_kernels=num_kernels, group=group,
                    filler=fillers.GaussianRandFiller())
                result = checker.check(layer, [input_blob], [output_blob])
                self.assertEqual(output_blob.data().shape[-1], num_kernels * group)
                print(result)
                self.assertTrue(result[0])
        # check if we will be able to produce an exception
        input_blob = base.Blob((1,5,5,3), filler=fillers.GaussianRandFiller())
        self.assertRaises(RuntimeError, checker.check,
                          layer, [input_blob], [output_blob])
        

if __name__ == '__main__':
    unittest.main()
